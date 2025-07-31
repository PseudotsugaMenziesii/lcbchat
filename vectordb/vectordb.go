package vectordb

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"lcbchat/embedding"
	"lcbchat/chunking"
)

// VectorDBConfig holds configuration for the vector database
type VectorDBConfig struct {
	Host           string
	Port           int
	CollectionName string
	VectorSize     int    // Dimension of your embeddings
	Distance       string // "Cosine", "Euclidean", or "Dot"
	IndexType      string // "HNSW" or "Plain"
}

// DefaultVectorDBConfig returns sensible defaults for Qdrant
func DefaultVectorDBConfig() VectorDBConfig {
	return VectorDBConfig{
		Host:           "localhost",
		Port:           6334, // Default Qdrant gRPC port
		CollectionName: "scraped_content",
		VectorSize:     768,      // Default for nomic-embed-text
		Distance:       "Cosine", // Good for semantic similarity
		IndexType:      "HNSW",   // Hierarchical Navigable Small World - fast and accurate
	}
}

// VectorDatabase wraps Qdrant operations
type VectorDatabase struct {
	collectionsClient qdrant.CollectionsClient
	pointsClient      qdrant.PointsClient
	conn              *grpc.ClientConn
	config            VectorDBConfig
	collection        string
}

// SearchResult represents a search result from the vector database
type SearchResult struct {
	Chunk     embedding.EmbeddedChunk
	Score     float32
	ID        string
	Metadata  map[string]interface{}
}

// NewVectorDatabase creates a new vector database connection
func NewVectorDatabase(config VectorDBConfig) (*VectorDatabase, error) {
	// Connect to Qdrant
	//conn, err := grpc.Dial(
	conn, err := grpc.NewClient(
		fmt.Sprintf("%s:%d", config.Host, config.Port),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Qdrant: %w", err)
	}

	// Create the specific clients we need
	collectionsClient := qdrant.NewCollectionsClient(conn)
	pointsClient := qdrant.NewPointsClient(conn)

	db := &VectorDatabase{
		collectionsClient: collectionsClient,
		pointsClient:      pointsClient,
		conn:              conn,
		config:            config,
		collection:        config.CollectionName,
	}

	return db, nil
}

// InitializeCollection creates the collection if it doesn't exist
func (vdb *VectorDatabase) InitializeCollection(ctx context.Context) error {
	// Check if collection exists
	collections, err := vdb.collectionsClient.List(ctx, &qdrant.ListCollectionsRequest{})
	if err != nil {
		return fmt.Errorf("failed to list collections: %w", err)
	}

	// Check if our collection already exists
	for _, collection := range collections.Collections {
		if collection.Name == vdb.collection {
			log.Printf("Collection '%s' already exists", vdb.collection)
			return nil
		}
	}

	// Create the collection
	log.Printf("Creating collection '%s' with vector size %d", vdb.collection, vdb.config.VectorSize)

	// Map distance types
	var distance qdrant.Distance
	switch vdb.config.Distance {
	case "Cosine":
		distance = qdrant.Distance_Cosine
	case "Euclidean":
		distance = qdrant.Distance_Euclid
	case "Dot":
		distance = qdrant.Distance_Dot
	default:
		distance = qdrant.Distance_Cosine
	}

	// Create collection with vector configuration
	_, err = vdb.collectionsClient.Create(ctx, &qdrant.CreateCollection{
		CollectionName: vdb.collection,
		VectorsConfig: &qdrant.VectorsConfig{
			Config: &qdrant.VectorsConfig_Params{
				Params: &qdrant.VectorParams{
					Size:     uint64(vdb.config.VectorSize),
					Distance: distance,
				},
			},
		},
		// Configure HNSW index for better performance
		HnswConfig: &qdrant.HnswConfigDiff{
			M:              uint64Ptr(uint64(16)), // Number of connections per node
			EfConstruct:    uint64Ptr(uint64(100)), // Size of candidate list during construction
			FullScanThreshold: uint64Ptr(uint64(10000)), // Use full scan for small collections
		},
		// Configure indexing
		OptimizersConfig: &qdrant.OptimizersConfigDiff{
			DefaultSegmentNumber: uint64Ptr(uint64(2)),
		},
	})

	if err != nil {
		return fmt.Errorf("failed to create collection: %w", err)
	}

	log.Printf("Successfully created collection '%s'", vdb.collection)
	return nil
}

// IndexEmbeddedChunks adds embedded chunks to the vector database
func (vdb *VectorDatabase) IndexEmbeddedChunks(ctx context.Context, chunks []embedding.EmbeddedChunk) error {
	if len(chunks) == 0 {
		return fmt.Errorf("no chunks to index")
	}

	log.Printf("Indexing %d chunks to collection '%s'", len(chunks), vdb.collection)

	// Convert chunks to Qdrant points
	points := make([]*qdrant.PointStruct, len(chunks))
	for i, chunk := range chunks {
		// Convert embedding to float32 (Qdrant requirement)
		vector := make([]float32, len(chunk.Embedding))
		for j, val := range chunk.Embedding {
			vector[j] = float32(val)
		}

		// Create payload with all chunk metadata
		payload := map[string]*qdrant.Value{
			"source_url":    qdrant.NewValueString(chunk.SourceURL),
			"title":         qdrant.NewValueString(chunk.Title),
			"content":       qdrant.NewValueString(chunk.Content),
			"chunk_index":   qdrant.NewValueInt(int64(chunk.ChunkIndex)),
			"word_count":    qdrant.NewValueInt(int64(chunk.WordCount)),
			"char_count":    qdrant.NewValueInt(int64(chunk.CharCount)),
			"content_type":  qdrant.NewValueString(chunk.Metadata["content_type"].(string)),
			"embedded_at":   qdrant.NewValueString(chunk.EmbeddedAt.Format(time.RFC3339)),
			"model_used":    qdrant.NewValueString(chunk.ModelUsed),
		}

		// Add timestamp if available
		if timestamp, ok := chunk.Metadata["timestamp"].(time.Time); ok {
			payload["timestamp"] = qdrant.NewValueString(timestamp.Format(time.RFC3339))
		}

		points[i] = &qdrant.PointStruct{
			Id: &qdrant.PointId{
				PointIdOptions: &qdrant.PointId_Uuid{
					Uuid: chunk.EmbeddingID,
				},
			},
			Vectors: &qdrant.Vectors{
				VectorsOptions: &qdrant.Vectors_Vector{
					Vector: &qdrant.Vector{Data: vector},
				},
			},
			Payload: payload,
		}
	}

	// Upsert points in batches for better performance
	batchSize := 100
	for i := 0; i < len(points); i += batchSize {
		end := i + batchSize
		if end > len(points) {
			end = len(points)
		}

		batch := points[i:end]
		_, err := vdb.pointsClient.Upsert(ctx, &qdrant.UpsertPoints{
			CollectionName: vdb.collection,
			Points:         batch,
		})

		if err != nil {
			return fmt.Errorf("failed to upsert batch %d-%d: %w", i, end, err)
		}

		log.Printf("Indexed batch %d-%d of %d chunks", i+1, end, len(points))
	}

	log.Printf("Successfully indexed all %d chunks", len(chunks))
	return nil
}

// Search performs semantic search in the vector database
func (vdb *VectorDatabase) Search(ctx context.Context, queryVector []float64, limit uint64, filters map[string]interface{}) ([]SearchResult, error) {
	// Convert query vector to float32
	vector := make([]float32, len(queryVector))
	for i, val := range queryVector {
		vector[i] = float32(val)
	}

	// Build filters if provided
	var filter *qdrant.Filter
	if len(filters) > 0 {
		conditions := make([]*qdrant.Condition, 0, len(filters))
		
		for key, value := range filters {
			switch v := value.(type) {
			case string:
				conditions = append(conditions, &qdrant.Condition{
					ConditionOneOf: &qdrant.Condition_Field{
						Field: &qdrant.FieldCondition{
							Key: key,
							Match: &qdrant.Match{
								MatchValue: &qdrant.Match_Keyword{
									Keyword: v,
								},
							},
						},
					},
				})
			case int, int64:
				val := int64(0)
				if i, ok := v.(int); ok {
					val = int64(i)
				} else {
					val = v.(int64)
				}
				conditions = append(conditions, &qdrant.Condition{
					ConditionOneOf: &qdrant.Condition_Field{
						Field: &qdrant.FieldCondition{
							Key: key,
							Match: &qdrant.Match{
								MatchValue: &qdrant.Match_Integer{
									Integer: val,
								},
							},
						},
					},
				})
			}
		}

		filter = &qdrant.Filter{
			Must: conditions,
		}
	}

	// Perform the search
	searchResult, err := vdb.pointsClient.Search(ctx, &qdrant.SearchPoints{
		CollectionName: vdb.collection,
		Vector:         vector,
		Limit:          limit,
		Filter:         filter,
		WithPayload:    &qdrant.WithPayloadSelector{SelectorOptions: &qdrant.WithPayloadSelector_Enable{Enable: true}},
		Params: &qdrant.SearchParams{
			HnswEf: uint64Ptr(uint64(128)), // Higher values = more accurate but slower
		},
	})

	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	// Convert results
	results := make([]SearchResult, len(searchResult.Result))
	for i, point := range searchResult.Result {
		// Extract payload values
		payload := make(map[string]interface{})
		for key, value := range point.Payload {
			switch v := value.Kind.(type) {
			case *qdrant.Value_StringValue:
				payload[key] = v.StringValue
			case *qdrant.Value_IntegerValue:
				payload[key] = v.IntegerValue
			case *qdrant.Value_DoubleValue:
				payload[key] = v.DoubleValue
			case *qdrant.Value_BoolValue:
				payload[key] = v.BoolValue
			}
		}

		// Reconstruct EmbeddedChunk (without embedding vector for efficiency)
		chunk := embedding.EmbeddedChunk{
			Chunk: chunking.Chunk{
				ID:         point.Id.GetUuid(),
				SourceURL:  payload["source_url"].(string),
				Title:      payload["title"].(string),
				Content:    payload["content"].(string),
				ChunkIndex: int(payload["chunk_index"].(int64)),
				WordCount:  int(payload["word_count"].(int64)),
				CharCount:  int(payload["char_count"].(int64)),
				Metadata: map[string]interface{}{
					"content_type": payload["content_type"],
				},
			},
			ModelUsed: payload["model_used"].(string),
		}

		// Parse timestamps
		if embeddedAt, ok := payload["embedded_at"].(string); ok {
			if t, err := time.Parse(time.RFC3339, embeddedAt); err == nil {
				chunk.EmbeddedAt = t
			}
		}
		if timestamp, ok := payload["timestamp"].(string); ok {
			if t, err := time.Parse(time.RFC3339, timestamp); err == nil {
				chunk.Metadata["timestamp"] = t
			}
		}

		results[i] = SearchResult{
			Chunk:    chunk,
			Score:    point.Score,
			ID:       point.Id.GetUuid(),
			Metadata: payload,
		}
	}

	return results, nil
}

// GetCollectionInfo returns information about the collection
func (vdb *VectorDatabase) GetCollectionInfo(ctx context.Context) (*qdrant.CollectionInfo, error) {
	collectionInfo, err := vdb.collectionsClient.Get(ctx, &qdrant.GetCollectionInfoRequest{
		CollectionName: vdb.collection,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get collection info: %w", err)
	}
	
	return collectionInfo.Result, nil
}

// DeleteCollection removes the entire collection
func (vdb *VectorDatabase) DeleteCollection(ctx context.Context) error {
	_, err := vdb.collectionsClient.Delete(ctx, &qdrant.DeleteCollection{
		CollectionName: vdb.collection,
	})
	return err
}

// Close closes the database connection
func (vdb *VectorDatabase) Close() error {
	return vdb.conn.Close()
}

func uint64Ptr(v uint64) *uint64 { return &v }

// LoadEmbeddedChunksFromJSON loads embedded chunks from a JSON file
func LoadEmbeddedChunksFromJSON(filepath string) ([]embedding.EmbeddedChunk, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file %s: %w", filepath, err)
	}

	var chunks []embedding.EmbeddedChunk
	if err := json.Unmarshal(data, &chunks); err != nil {
		return nil, fmt.Errorf("failed to unmarshal embedded chunks: %w", err)
	}

	return chunks, nil
}

// SetupVectorDatabase initializes the vector database and indexes chunks
func SetupVectorDatabase(embeddedChunksFile string, config VectorDBConfig) (*VectorDatabase, error) {
	ctx := context.Background()

	// Load embedded chunks
	log.Printf("Loading embedded chunks from %s", embeddedChunksFile)
	chunks, err := LoadEmbeddedChunksFromJSON(embeddedChunksFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load embedded chunks: %w", err)
	}

	if len(chunks) == 0 {
		return nil, fmt.Errorf("no embedded chunks found in file")
	}

	// Auto-detect vector size from first chunk
	if config.VectorSize == 0 {
		config.VectorSize = len(chunks[0].Embedding)
		log.Printf("Auto-detected vector size: %d", config.VectorSize)
	}

	// Connect to vector database
	vdb, err := NewVectorDatabase(config)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to vector database: %w", err)
	}

	// Initialize collection
	if err := vdb.InitializeCollection(ctx); err != nil {
		return nil, fmt.Errorf("failed to initialize collection: %w", err)
	}

	// Index the chunks
	if err := vdb.IndexEmbeddedChunks(ctx, chunks); err != nil {
		return nil, fmt.Errorf("failed to index chunks: %w", err)
	}

	// Verify the indexing
	info, err := vdb.GetCollectionInfo(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get collection info: %w", err)
	}

	log.Printf("Vector database setup complete. Collection '%s' has %d points", 
		config.CollectionName, info.PointsCount)

	return vdb, nil
}