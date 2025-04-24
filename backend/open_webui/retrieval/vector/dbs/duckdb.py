from typing import Optional, List, Union, Dict, Any
import logging
import duckdb
import json
import os

from open_webui.retrieval.vector.main import VectorItem, SearchResult, GetResult
from open_webui.config import DUCKDB_PATH
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


class DuckDBClient:
    def __init__(self):
        """
        Initialize the DuckDB client.
        
        The database file path is taken from DUCKDB_PATH config variable.
        If not provided, it defaults to 'vector_store.db' in the current directory.
        """
        self.db_path = DUCKDB_PATH or os.path.join(os.getcwd(), "vector_store.db")
        self.collection_prefix = "open_webui"
        
        # Connect to the database
        self.conn = duckdb.connect(self.db_path)


        
        # Enable the VSS extension (Vector Similarity Search)
        try:
            self.conn.execute("INSTALL vss; LOAD vss;")
            self.vss_extension_available = True
            self.conn.execute("SET hnsw_enable_experimental_persistence=true;")
            log.info("DuckDB VSS (Vector Similarity Search) extension loaded successfully.")
        except Exception as e:
            self.vss_extension_available = False
            log.warning(f"DuckDB VSS extension not available: {e}. Falling back to manual cosine similarity.")
        
        # Initialize database schema
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the database schema with required tables."""
        try:
            # Create metadata table for collections
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    name VARCHAR PRIMARY KEY,
                    dimension INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            log.info("Database initialized successfully.")
        except Exception as e:
            log.exception(f"Error initializing database: {e}")
            raise

    def _get_collection_table_name(self, collection_name: str) -> str:
        """Generate the internal table name for a collection."""
        # Replace hyphens with underscores for SQL compatibility
        safe_name = collection_name.replace("-", "_")
        return f"{self.collection_prefix}_{safe_name}"
    
    def _create_collection_table(self, collection_name: str, dimension: int):
        """Create a new collection table with the given name and vector dimension."""
        table_name = self._get_collection_table_name(collection_name)
        
        try:
            # Create the collection table
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id VARCHAR PRIMARY KEY,
                    vector FLOAT[{dimension}] NOT NULL,
                    text TEXT,
                    metadata JSON
                )
            """)
            
            # Create an HNSW index on the vector column if VSS is available
            if self.vss_extension_available:
                try:
                    self.conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{table_name}_vector 
                        ON {table_name} USING HNSW(vector)
                    """)
                    log.info(f"Created HNSW index for collection '{collection_name}'")
                except Exception as e:
                    log.warning(f"Failed to create HNSW index for collection '{collection_name}': {e}")
                    self.vss_extension_available = False
            
            # Register the collection in the metadata table
            self.conn.execute(f"""
                INSERT INTO collections (name, dimension) 
                VALUES (?, ?)
                ON CONFLICT (name) DO UPDATE SET dimension = excluded.dimension
            """, [collection_name, dimension])
            
            log.info(f"Created collection '{collection_name}' with dimension {dimension}")
            return True
        except Exception as e:
            log.exception(f"Error creating collection '{collection_name}': {e}")
            return False

    def _calculate_cosine_similarity(self, v1, v2) -> float:
        """Calculate cosine similarity between two vectors."""
        if self.vss_extension_available:
            # Use the VSS extension if available (better performance)
            return self.conn.execute(f"SELECT vss_cosine_similarity(?, ?)", [v1, v2]).fetchone()[0]
        else:
            # Manual calculation as fallback
            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm_v1 = sum(a * a for a in v1) ** 0.5
            norm_v2 = sum(b * b for b in v2) ** 0.5
            if norm_v1 == 0 or norm_v2 == 0:
                return 0
            return dot_product / (norm_v1 * norm_v2)

    def has_collection(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            result = self.conn.execute(
                "SELECT COUNT(*) FROM collections WHERE name = ?",
                [collection_name]
            ).fetchone()[0]
            return result > 0
        except Exception as e:
            log.exception(f"Error checking collection existence: {e}")
            return False

    def delete_collection(self, collection_name: str):
        """Delete a collection and its associated data."""
        if not self.has_collection(collection_name):
            log.warning(f"Collection '{collection_name}' does not exist.")
            return
        
        table_name = self._get_collection_table_name(collection_name)
        
        try:
            # Drop the collection table
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Remove from collections metadata
            self.conn.execute("DELETE FROM collections WHERE name = ?", [collection_name])
            
            log.info(f"Collection '{collection_name}' deleted successfully.")
        except Exception as e:
            log.exception(f"Error deleting collection '{collection_name}': {e}")
            raise

    def search(
        self, 
        collection_name: str, 
        vectors: List[List[Union[float, int]]], 
        limit: int
    ) -> Optional[SearchResult]:
        """Search for vectors by similarity."""
        if not self.has_collection(collection_name):
            log.warning(f"Collection '{collection_name}' does not exist.")
            return None
        
        if not vectors:
            log.warning("No search vectors provided.")
            return None
        
        table_name = self._get_collection_table_name(collection_name)
        
        try:
            # Get the collection's dimension
            dimension = self.conn.execute(
                "SELECT dimension FROM collections WHERE name = ?", 
                [collection_name]
            ).fetchone()[0]
            
            # For each query vector, find the most similar vectors
            all_ids = []
            all_documents = []
            all_metadatas = []
            all_distances = []
            
            for query_vector in vectors:
                # Ensure the query vector has the correct dimension
                if len(query_vector) != dimension:
                    query_vector = query_vector[:dimension] if len(query_vector) > dimension else query_vector + [0] * (dimension - len(query_vector))
                
                if self.vss_extension_available:
                    try:
                        # Use HNSW index for efficient similarity search (cosine distance)
                        # Cast to proper type to ensure index usage
                        typed_vector = f"{query_vector}::FLOAT[{dimension}]"
                        query_result = self.conn.execute(f"""
                            SELECT id, text, metadata, 
                                1 - array_cosine_distance(vector, {typed_vector}) AS similarity
                            FROM {table_name} 
                            ORDER BY array_cosine_distance(vector, {typed_vector})
                            LIMIT ?
                        """, [limit]).fetchall()
                        
                        # Alternative using min_by if the above doesn't work
                        if not query_result:
                            query_result = self.conn.execute(f"""
                                SELECT min_by({table_name}, array_cosine_distance(vector, {typed_vector}), ? ORDER BY vector) AS result
                                FROM {table_name}
                            """, [limit]).fetchall()
                    except Exception as e:
                        log.warning(f"HNSW query failed, falling back to manual calculation: {e}")
                        self.vss_extension_available = False
                
                if not self.vss_extension_available:
                    # Fetch all vectors and calculate similarity in Python (less efficient)
                    all_vectors = self.conn.execute(f"""
                        SELECT id, text, metadata, vector 
                        FROM {table_name}
                    """).fetchall()
                    
                    # Calculate similarities
                    similarities = []
                    for row in all_vectors:
                        similarity = self._calculate_cosine_similarity(query_vector, row[3])
                        similarities.append((row[0], row[1], row[2], similarity))
                    
                    # Sort by similarity and take top k
                    similarities.sort(key=lambda x: x[3], reverse=True)
                    query_result = similarities[:limit]
                
                # Extract results
                ids = []
                documents = []
                metadatas = []
                distances = []
                
                for row in query_result:
                    ids.append(row[0])
                    documents.append(row[1])
                    metadatas.append(json.loads(row[2]) if isinstance(row[2], str) else row[2])
                    distances.append(row[3])
                
                all_ids.append(ids)
                all_documents.append(documents)
                all_metadatas.append(metadatas)
                all_distances.append(distances)
            
            return SearchResult(
                ids=all_ids,
                documents=all_documents,
                metadatas=all_metadatas,
                distances=all_distances,
            )
        except Exception as e:
            log.exception(f"Error searching collection '{collection_name}': {e}")
            return None
    
    def query(
        self, 
        collection_name: str, 
        filter: Dict[str, Any], 
        limit: Optional[int] = None
    ) -> Optional[GetResult]:
        """Query items from a collection based on metadata filters."""
        if not self.has_collection(collection_name):
            log.warning(f"Collection '{collection_name}' does not exist.")
            return None
        
        table_name = self._get_collection_table_name(collection_name)
        
        try:
            # Build the query
            query = f"SELECT id, text, metadata FROM {table_name}"
            params = []
            
            # Add filter conditions if provided
            if filter:
                conditions = []
                for key, value in filter.items():
                    # For JSON fields, we need to use the JSON extraction syntax
                    conditions.append(f"json_extract_string(metadata, '$.{key}') = ?")
                    params.append(str(value))
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            # Add limit if provided
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            
            # Execute the query
            results = self.conn.execute(query, params).fetchall()
            
            if not results:
                return None
            
            # Process results
            ids = []
            documents = []
            metadatas = []
            
            for row in results:
                ids.append(row[0])
                documents.append(row[1])
                metadatas.append(json.loads(row[2]) if isinstance(row[2], str) else row[2])
            
            return GetResult(
                ids=[ids],
                documents=[documents],
                metadatas=[metadatas],
            )
        except Exception as e:
            log.exception(f"Error querying collection '{collection_name}': {e}")
            return None

    def get(self, collection_name: str) -> Optional[GetResult]:
        """Get all items in a collection."""
        if not self.has_collection(collection_name):
            log.warning(f"Collection '{collection_name}' does not exist.")
            return None
        
        table_name = self._get_collection_table_name(collection_name)
        
        try:
            results = self.conn.execute(f"SELECT id, text, metadata FROM {table_name}").fetchall()
            
            if not results:
                return None
            
            ids = []
            documents = []
            metadatas = []
            
            for row in results:
                ids.append(row[0])
                documents.append(row[1])
                metadatas.append(json.loads(row[2]) if isinstance(row[2], str) else row[2])
            
            return GetResult(
                ids=[ids],
                documents=[documents],
                metadatas=[metadatas],
            )
        except Exception as e:
            log.exception(f"Error getting items from collection '{collection_name}': {e}")
            return None

    def insert(self, collection_name: str, items: List[VectorItem]):
        """Insert items into a collection."""
        if not items:
            log.warning("No items to insert.")
            return
        
        # Get the dimension from the first item
        dimension = len(items[0]["vector"])
        table_name = self._get_collection_table_name(collection_name)
        
        try:
            # Create collection if it doesn't exist
            if not self.has_collection(collection_name):
                self._create_collection_table(collection_name, dimension)
            
            # Insert items
            for item in items:
                vector = item["vector"]
                # Ensure vector has correct dimension
                if len(vector) != dimension:
                    log.warning(f"Vector dimension mismatch. Expected {dimension}, got {len(vector)}. Adjusting...")
                    vector = vector[:dimension] if len(vector) > dimension else vector + [0] * (dimension - len(vector))
                
                # Insert the item
                self.conn.execute(f"""
                    INSERT INTO {table_name} (id, vector, text, metadata)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (id) DO NOTHING
                """, [
                    item["id"],
                    vector,
                    item["text"],
                    json.dumps(item["metadata"]) if item["metadata"] else None
                ])
            
            log.info(f"Inserted {len(items)} items into collection '{collection_name}'")
        except Exception as e:
            log.exception(f"Error inserting items into collection '{collection_name}': {e}")
            raise

    def upsert(self, collection_name: str, items: List[VectorItem]):
        """Update existing items or insert new ones."""
        if not items:
            log.warning("No items to upsert.")
            return
        
        # Get the dimension from the first item
        dimension = len(items[0]["vector"])
        table_name = self._get_collection_table_name(collection_name)
        
        try:
            # Create collection if it doesn't exist
            if not self.has_collection(collection_name):
                self._create_collection_table(collection_name, dimension)
            
            # Upsert items
            for item in items:
                vector = item["vector"]
                # Ensure vector has correct dimension
                if len(vector) != dimension:
                    log.warning(f"Vector dimension mismatch. Expected {dimension}, got {len(vector)}. Adjusting...")
                    vector = vector[:dimension] if len(vector) > dimension else vector + [0] * (dimension - len(vector))
                
                # Upsert the item
                self.conn.execute(f"""
                    INSERT INTO {table_name} (id, vector, text, metadata)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (id) DO UPDATE SET
                        vector = excluded.vector,
                        text = excluded.text,
                        metadata = excluded.metadata
                """, [
                    item["id"],
                    vector,
                    item["text"],
                    json.dumps(item["metadata"]) if item["metadata"] else None
                ])
            
            log.info(f"Upserted {len(items)} items in collection '{collection_name}'")
        except Exception as e:
            log.exception(f"Error upserting items in collection '{collection_name}': {e}")
            raise

    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None
    ):
        """Delete items from a collection by ID or filter criteria."""
        if not self.has_collection(collection_name):
            log.warning(f"Collection '{collection_name}' does not exist.")
            return
        
        if not ids and not filter:
            log.warning("No deletion criteria provided (ids or filter).")
            return
        
        table_name = self._get_collection_table_name(collection_name)
        
        try:
            if ids:
                # Delete by IDs
                placeholders = ', '.join(['?'] * len(ids))
                self.conn.execute(f"DELETE FROM {table_name} WHERE id IN ({placeholders})", ids)
                log.info(f"Deleted items with IDs {ids} from collection '{collection_name}'")
            
            if filter:
                # Delete by filter
                conditions = []
                params = []
                for key, value in filter.items():
                    conditions.append(f"json_extract_string(metadata, '$.{key}') = ?")
                    params.append(str(value))
                
                if conditions:
                    query = f"DELETE FROM {table_name} WHERE " + " AND ".join(conditions)
                    self.conn.execute(query, params)
                    log.info(f"Deleted items with filter {filter} from collection '{collection_name}'")
        except Exception as e:
            log.exception(f"Error deleting items from collection '{collection_name}': {e}")
            raise