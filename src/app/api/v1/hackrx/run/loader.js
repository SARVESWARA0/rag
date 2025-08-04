import { Pinecone } from '@pinecone-database/pinecone';
import { createMistral } from '@ai-sdk/mistral';
import { embedMany } from 'ai';
import fs from 'fs/promises';
import path from 'path';

const config = {
    indexName: 'rag',
    dimension: 1024, // Updated to 1024 for Mistral embeddings
    batchSize: 10,
    recordsPerNamespace: 300,
    wordsPerChunk: 350
};
const mistral = createMistral({ apiKey: process.env.MISTRAL_API_KEY });
async function initServices() {
    try {
        const pinecone = new Pinecone({
            apiKey: process.env.PINECONE_API_KEY,
        });

        const index = pinecone.index(config.indexName);

        return { pinecone, index };
    } catch (error) {
        console.error("Error initializing services:", error);
        throw error;
    }
}

function splitTextIntoChunks(
  text,
  maxTokens = 350,
  overlapTokens = 50,
  tokenize = str => str.split(/\s+/)  // replace with a real tokenizer if you have one
) {
  if (!text.trim()) return [];

  // Break into sentences
  const sentences = text.match(/[^\.!\?]+[\.!\?]+(?:\s|$)/g) || [text];
  const chunks = [];
  let currentTokens = [];

  for (const sentence of sentences) {
    const sentTokens = tokenize(sentence);
    if (currentTokens.length + sentTokens.length > maxTokens) {
      // flush current chunk
      chunks.push(currentTokens.join(' '));
      // carry over overlap
      currentTokens = currentTokens.slice(-overlapTokens);
    }
    currentTokens.push(...sentTokens);
  }
  if (currentTokens.length) chunks.push(currentTokens.join(' '));

  return chunks.map(c => ({
    content: c,
    wordCount: c.split(/\s+/).length
  }));
}


async function readAndProcessFile(filePath) {
    try {
        const content = await fs.readFile(filePath, 'utf8');
        
        if (!content || content.trim().length === 0) {
            throw new Error('File is empty or contains only whitespace');
        }

        return splitTextIntoChunks(content);
    } catch (error) {
        console.error(`Error processing file ${filePath}:`, error);
        throw error;
    }
}

async function generateEmbeddings(chunks) {
    try {
        console.log(`Generating embeddings for ${chunks.length} chunks`);
        const embeddings = [];
        const batchSize = 10; // Increased batch size for better efficiency
        
        for (let i = 0; i < chunks.length; i += batchSize) {
            const batch = chunks.slice(i, i + batchSize);
            console.log(`Processing batch ${Math.floor(i/batchSize) + 1}, chunks ${i+1}-${Math.min(i+batchSize, chunks.length)}`);
            
            // Extract text content from chunks
            const texts = batch.map(chunk => chunk.content);
            
            // Generate embeddings using Mistral model
            const { embeddings: batchEmbeddings } = await embedMany({
                model: mistral.textEmbeddingModel('mistral-embed'),
                values: texts,
            });
            
            // Combine embeddings with chunk metadata
            const results = batchEmbeddings.map((embedding, index) => ({
                embedding: embedding,
                content: batch[index].content,
                wordCount: batch[index].wordCount
            }));
            
            console.log(`Generated embeddings for batch ${Math.floor(i/batchSize) + 1}, embedding length: ${results[0].embedding.length}`);
            embeddings.push(...results);
        }
        console.log(`Total embeddings generated: ${embeddings.length}`);
        return embeddings;
    } catch (error) {
        console.error("Error in batch embedding generation:", error);
        throw error;
    }
}

async function getCurrentNamespaceCount(index, namespace) {
    try {
        const stats = await index.describeIndexStats({
            filter: { namespace: namespace }
        });
        console.log(`Namespace ${namespace} stats:`, stats.namespaces[namespace]);
        return stats.namespaces[namespace]?.recordCount || 0;
    } catch (error) {
        console.error(`Error getting namespace count for ${namespace}:`, error);
        return 0;
    }
}

async function getNextNamespace(index, baseNamespace = 'default') {
    let namespaceIndex = 1;
    let currentNamespace = baseNamespace;
    
    while (true) {
        const count = await getCurrentNamespaceCount(index, currentNamespace);
        
        if (count < config.recordsPerNamespace) {
            return { namespace: currentNamespace, currentCount: count };
        }
        
        namespaceIndex++;
        currentNamespace = `${baseNamespace}_${namespaceIndex}`;
    }
}

async function processAndUpsert(chunks, fileName, index) {
    const results = new Map();
    const embeddingsWithMetadata = await generateEmbeddings(chunks);
    let currentBatch = [];
    let currentNamespaceInfo = await getNextNamespace(index);
    let processedCount = 0;
    
    console.log(`Starting with namespace: ${currentNamespaceInfo.namespace}`);
    
    for (const embedData of embeddingsWithMetadata) {
   
        if (currentNamespaceInfo.currentCount >= config.recordsPerNamespace) {
           
            if (currentBatch.length > 0) {
                await index.namespace(currentNamespaceInfo.namespace).upsert(currentBatch);
            }
            
            currentNamespaceInfo = await getNextNamespace(index);
            currentBatch = [];
            console.log(`Switching to namespace: ${currentNamespaceInfo.namespace}`);
        }

        const record = {
            id: `${fileName.replace(/[^a-zA-Z0-9]/g, '_')}_chunk_${processedCount}`,
            values: embedData.embedding,
            metadata: {
                fileName,
                content: embedData.content,
                wordCount: embedData.wordCount,
                namespace: currentNamespaceInfo.namespace,
                timestamp: new Date().toISOString()
            }
        };

        currentBatch.push(record);
        currentNamespaceInfo.currentCount++;
        processedCount++;

        // Track results
        if (!results.has(currentNamespaceInfo.namespace)) {
            results.set(currentNamespaceInfo.namespace, 0);
        }
        results.set(
            currentNamespaceInfo.namespace, 
            results.get(currentNamespaceInfo.namespace) + 1
        );

            if (currentBatch.length >= config.batchSize) {
        try {
            console.log(`Upserting batch of ${currentBatch.length} records to namespace ${currentNamespaceInfo.namespace}`);
            console.log(`First record sample:`, {
                id: currentBatch[0].id,
                contentLength: currentBatch[0].metadata.content.length,
                embeddingLength: currentBatch[0].values.length
            });
            await index.namespace(currentNamespaceInfo.namespace).upsert(currentBatch);
            console.log(`Successfully processed batch of ${currentBatch.length} records in namespace ${currentNamespaceInfo.namespace}`);
            currentBatch = [];
        } catch (error) {
            console.error(`Error upserting batch to namespace ${currentNamespaceInfo.namespace}:`, error);
            throw error;
        }
    }
    }

    if (currentBatch.length > 0) {
        try {
            console.log(`Upserting final batch of ${currentBatch.length} records to namespace ${currentNamespaceInfo.namespace}`);
            console.log(`Final record sample:`, {
                id: currentBatch[0].id,
                contentLength: currentBatch[0].metadata.content.length,
                embeddingLength: currentBatch[0].values.length
            });
            await index.namespace(currentNamespaceInfo.namespace).upsert(currentBatch);
            console.log(`Successfully processed final batch of ${currentBatch.length} records in namespace ${currentNamespaceInfo.namespace}`);
        } catch (error) {
            console.error(`Error upserting final batch to namespace ${currentNamespaceInfo.namespace}:`, error);
            throw error;
        }
    }

    return results;
}

// Modified main function to work with API routes
async function main(customFilePath = null) {
    try {
        const { pinecone, index } = await initServices();
        
        // Use custom file path if provided, otherwise default to './data.txt'
        const filePath = customFilePath || './data.txt';
        const fileName = path.basename(filePath);
        
        console.log(`\nProcessing file: ${filePath}`);
        
        const chunks = await readAndProcessFile(filePath);
        const results = await processAndUpsert(chunks, fileName, index);
        
        console.log('\nProcessing results:');
        const resultObj = {};
        for (const [namespace, count] of results.entries()) {
            console.log(`${namespace}: ${count} chunks processed`);
            resultObj[namespace] = count;
        }

        const stats = await index.describeIndexStats();
        console.log('\nFinal index stats:', JSON.stringify(stats, null, 2));

        // Return results for API response
        return {
            processedChunks: chunks.length,
            namespaces: resultObj,
            totalVectors: Object.values(resultObj).reduce((sum, count) => sum + count, 0),
            indexStats: stats
        };

    } catch (error) {
        console.error('Error in main process:', error);
        throw error; // Re-throw for API error handling
    }
}

// For standalone execution
process.on('unhandledRejection', (error) => {
    console.error('Unhandled promise rejection:', error);
    process.exit(1);
});

// Only run main if this file is executed directly (not imported)
if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}

export {
    initServices,
    readAndProcessFile,
    processAndUpsert,
    main
};