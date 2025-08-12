package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"lcbchat/chunking"
	"github.com/google/uuid"
)

// EmbeddedChunk represents a chunk with its embedding vector
type EmbeddedChunk struct {
	chunking.Chunk
	Embedding   []float64 `json:"embedding"`
	EmbeddedAt  time.Time `json:"embedded_at"`
	ModelUsed   string    `json:"model_used"`
	EmbeddingID string    `json:"embedding_id"`
}

// OllamaEmbeddingClient handles communication with Ollama for embeddings
type OllamaEmbeddingClient struct {
	BaseURL    string
	HTTPClient *http.Client
	Model      string
}

// OllamaEmbeddingRequest represents the request structure for Ollama embeddings
type OllamaEmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

// OllamaEmbeddingResponse represents the response structure from Ollama
type OllamaEmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

// EmbeddingConfig holds configuration for embedding generation
type EmbeddingConfig struct {
	Model           string        // Ollama model to use
	BatchSize       int           // Number of chunks to process in parallel
	RetryAttempts   int           // Number of retry attempts for failed embeddings
	RetryDelay      time.Duration // Delay between retries
	RequestTimeout  time.Duration // Timeout for individual requests
}

// DefaultEmbeddingConfig returns sensible defaults
func DefaultEmbeddingConfig() EmbeddingConfig {
	return EmbeddingConfig{
		Model:          "nomic-embed-text", // Good general-purpose embedding model
		BatchSize:      5,
		RetryAttempts:  3,
		RetryDelay:     2 * time.Second,
		RequestTimeout: 30 * time.Second,
	}
}

// NewOllamaEmbeddingClient creates a new Ollama embedding client
func NewOllamaEmbeddingClient(baseURL, model string) *OllamaEmbeddingClient {
	return &OllamaEmbeddingClient{
		BaseURL: baseURL,
		Model:   model,
		HTTPClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

// GenerateEmbedding generates an embedding for a single text
func (c *OllamaEmbeddingClient) GenerateEmbedding(ctx context.Context, text string) ([]float64, error) {
	request := OllamaEmbeddingRequest{
		Model:  c.Model,
		Prompt: text,
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.BaseURL+"/api/embeddings", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, string(body))
	}

	var response OllamaEmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(response.Embedding) == 0 {
		return nil, fmt.Errorf("received empty embedding")
	}

	return response.Embedding, nil
}

// EmbeddingGenerator orchestrates the embedding generation process
type EmbeddingGenerator struct {
	client *OllamaEmbeddingClient
	config EmbeddingConfig
}

// NewEmbeddingGenerator creates a new embedding generator
func NewEmbeddingGenerator(ollamaURL string, config EmbeddingConfig) *EmbeddingGenerator {
	client := NewOllamaEmbeddingClient(ollamaURL, config.Model)
	client.HTTPClient.Timeout = config.RequestTimeout

	return &EmbeddingGenerator{
		client: client,
		config: config,
	}
}

// GenerateEmbeddings processes chunks and generates embeddings
func (eg *EmbeddingGenerator) GenerateEmbeddings(chunks []chunking.Chunk) ([]EmbeddedChunk, error) {
	log.Printf("Generating embeddings for %d chunks using model %s", len(chunks), eg.config.Model)

	embeddedChunks := make([]EmbeddedChunk, len(chunks))
	
	// Use a semaphore to limit concurrent requests
	semaphore := make(chan struct{}, eg.config.BatchSize)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var errors []error

	for i, chunk := range chunks {
		wg.Add(1)
		go func(index int, c chunking.Chunk) {
			defer wg.Done()
			
			// Acquire semaphore
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			embeddedChunk, err := eg.processChunk(c)
			if err != nil {
				mu.Lock()
				errors = append(errors, fmt.Errorf("failed to process chunk %s: %w", c.ID, err))
				mu.Unlock()
				return
			}

			mu.Lock()
			embeddedChunks[index] = embeddedChunk
			mu.Unlock()

			if (index+1)%10 == 0 || index == len(chunks)-1 {
				log.Printf("Processed %d/%d chunks", index+1, len(chunks))
			}
		}(i, chunk)
	}

	wg.Wait()

	if len(errors) > 0 {
		log.Printf("Encountered %d errors during embedding generation", len(errors))
		for _, err := range errors {
			log.Printf("Error: %v", err)
		}
		return nil, fmt.Errorf("failed to generate embeddings: %d errors occurred", len(errors))
	}

	log.Printf("Successfully generated embeddings for all %d chunks", len(chunks))
	return embeddedChunks, nil
}

// processChunk generates embedding for a single chunk with retry logic
func (eg *EmbeddingGenerator) processChunk(chunk chunking.Chunk) (EmbeddedChunk, error) {
	var embedding []float64
	var lastErr error

	for attempt := 0; attempt <= eg.config.RetryAttempts; attempt++ {
		if attempt > 0 {
			time.Sleep(eg.config.RetryDelay)
			log.Printf("Retrying chunk %s (attempt %d/%d)", chunk.ID, attempt, eg.config.RetryAttempts)
		}

		ctx, cancel := context.WithTimeout(context.Background(), eg.config.RequestTimeout)
		
		var err error
		embedding, err = eg.client.GenerateEmbedding(ctx, chunk.Content)
		cancel()

		if err == nil {
			break // Success
		}

		lastErr = err
	}
	if embedding == nil {
		return EmbeddedChunk{}, fmt.Errorf("failed after %d attempts: %w", eg.config.RetryAttempts+1, lastErr)
	}
	// Random seed UUID to make the same chucks generate the same embedding ID
	seedUuid := uuid.MustParse("5955ff11-0749-4f38-9cf9-60495cbfadf6")
	embeddingId := uuid.NewSHA1(seedUuid, []byte(chunk.ID + "_" + eg.config.Model)).String()

	return EmbeddedChunk{
		Chunk:       chunk,
		Embedding:   embedding,
		EmbeddedAt:  time.Now(),
		ModelUsed:   eg.config.Model,
		EmbeddingID: embeddingId,
	}, nil
}

// LoadChunksFromJSON loads chunks from a JSON file
func LoadChunksFromJSON(filepath string) ([]chunking.Chunk, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file %s: %w", filepath, err)
	}

	var chunks []chunking.Chunk
	if err := json.Unmarshal(data, &chunks); err != nil {
		return nil, fmt.Errorf("failed to unmarshal chunks: %w", err)
	}

	return chunks, nil
}

// SaveEmbeddedChunks saves embedded chunks to JSON files
func SaveEmbeddedChunks(embeddedChunks []EmbeddedChunk, outputDir string) error {
	if len(embeddedChunks) == 0 {
		return fmt.Errorf("no embedded chunks to save")
	}

	// Save all embedded chunks to a single file
	allEmbeddingsFile := filepath.Join(outputDir, "embedded_chunks_all.json")
	if err := saveEmbeddedChunksToJSON(embeddedChunks, allEmbeddingsFile); err != nil {
		return fmt.Errorf("failed to save all embedded chunks: %w", err)
	}

	// Group by source URL and save separately
	chunksBySource := make(map[string][]EmbeddedChunk)
	for _, chunk := range embeddedChunks {
		chunksBySource[chunk.SourceURL] = append(chunksBySource[chunk.SourceURL], chunk)
	}

	for url, urlChunks := range chunksBySource {
		hash := fmt.Sprintf("%x", []byte(url))[:8]
		filename := fmt.Sprintf("embedded_chunks_%s.json", hash)
		filepath := filepath.Join(outputDir, filename)

		if err := saveEmbeddedChunksToJSON(urlChunks, filepath); err != nil {
			log.Printf("Failed to save embedded chunks for %s: %v", url, err)
		}
	}

	log.Printf("Saved %d embedded chunks to %s", len(embeddedChunks), outputDir)
	return nil
}

// saveEmbeddedChunksToJSON saves embedded chunks to a JSON file
func saveEmbeddedChunksToJSON(chunks []EmbeddedChunk, filepath string) error {
	data, err := json.MarshalIndent(chunks, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal embedded chunks: %w", err)
	}

	return os.WriteFile(filepath, data, 0644)
}

// ValidateEmbeddings checks the quality of generated embeddings
func ValidateEmbeddings(embeddedChunks []EmbeddedChunk) error {
	if len(embeddedChunks) == 0 {
		return fmt.Errorf("no embedded chunks to validate")
	}

	expectedDimensions := len(embeddedChunks[0].Embedding)
	if expectedDimensions == 0 {
		return fmt.Errorf("first chunk has empty embedding")
	}

	for _, chunk := range embeddedChunks {
		if len(chunk.Embedding) == 0 {
			return fmt.Errorf("chunk %s has empty embedding", chunk.ID)
		}
		
		if len(chunk.Embedding) != expectedDimensions {
			return fmt.Errorf("chunk %s has embedding dimension %d, expected %d", 
				chunk.ID, len(chunk.Embedding), expectedDimensions)
		}

		// Check for NaN or infinite values
		for j, val := range chunk.Embedding {
			if math.IsNaN(val) {
				return fmt.Errorf("chunk %s has NaN value at position %d", chunk.ID, j)
			}
			if math.IsInf(val, 0) {
				return fmt.Errorf("chunk %s has infinite value at position %d", chunk.ID, j)
			}
		}
	}

	log.Printf("Validation passed: %d chunks with %d-dimensional embeddings", 
		len(embeddedChunks), expectedDimensions)
	return nil
}

// ProcessChunksToEmbeddings is a convenience function that loads chunks and generates embeddings
func ProcessChunksToEmbeddings(chunksFile, outputDir, ollamaURL string, config EmbeddingConfig) error {
	// Load chunks
	log.Printf("Loading chunks from %s", chunksFile)
	chunks, err := LoadChunksFromJSON(chunksFile)
	if err != nil {
		return fmt.Errorf("failed to load chunks: %w", err)
	}

	// Generate embeddings
	generator := NewEmbeddingGenerator(ollamaURL, config)
	embeddedChunks, err := generator.GenerateEmbeddings(chunks)
	if err != nil {
		return fmt.Errorf("failed to generate embeddings: %w", err)
	}

	// Validate embeddings
	if err := ValidateEmbeddings(embeddedChunks); err != nil {
		return fmt.Errorf("embedding validation failed: %w", err)
	}

	// Save embedded chunks
	err = os.MkdirAll(outputDir, 0755)
	if err != nil {
		return fmt.Errorf("failed to create output directory %s: %w", outputDir, err)
	}
	if err := SaveEmbeddedChunks(embeddedChunks, outputDir); err != nil {
		return fmt.Errorf("failed to save embedded chunks: %w", err)
	}

	return nil
}