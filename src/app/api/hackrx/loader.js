import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenerativeAI } from "@google/generative-ai";
import fs from 'fs/promises';
import path from 'path';

const config = {
    indexName: 'rag',
    dimension: 768,
    batchSize: 10,
    recordsPerNamespace: 300,
    wordsPerChunk: 350
};

async function initServices() {
    try {
        const pinecone = new Pinecone({
            apiKey: process.env.PINECONE_API_KEY,
        });

        const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
        const embeddingModel = genAI.getGenerativeModel({ model: "text-embedding-004" });
        const index = pinecone.index(config.indexName);

        return { pinecone, embeddingModel, index };
    } catch (error) {
        console.error("Error initializing services:", error);
        throw error;
    }
}

function splitTextIntoChunks(text) {
    if (!text || text.trim().length === 0) {
        console.log("Received empty or whitespace-only text");
        return [];
    }

    // First, split by headers (markdown headers)
    const headerSplit = text.split(/(?=^#+\s)/m);
    const chunks = [];
    
    for (const section of headerSplit) {
        if (!section.trim()) continue;
        
        // If section is too long, split it further
        if (section.length > 2000) {
            // Split by sentences while preserving context
            const sentences = section.split(/(?<=[.!?])\s+/);
            let currentChunk = '';
            
            for (const sentence of sentences) {
                if ((currentChunk + sentence).length > 1500) {
                    if (currentChunk.trim()) {
                        chunks.push({
                            content: currentChunk.trim(),
                            wordCount: currentChunk.split(/\s+/).length
                        });
                    }
                    currentChunk = sentence;
                } else {
                    currentChunk += (currentChunk ? ' ' : '') + sentence;
                }
            }
            
            if (currentChunk.trim()) {
                chunks.push({
                    content: currentChunk.trim(),
                    wordCount: currentChunk.split(/\s+/).length
                });
            }
        } else {
            // Section is appropriately sized
            chunks.push({
                content: section.trim(),
                wordCount: section.split(/\s+/).length
            });
        }
    }
    
    // Filter out very short chunks that might not be meaningful
    return chunks.filter(chunk => chunk.content.length > 50);
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

async function generateEmbeddings(chunks, embeddingModel) {
    try {
        const embeddings = [];
        const batchSize = 5; 
        for (let i = 0; i < chunks.length; i += batchSize) {
            const batch = chunks.slice(i, i + batchSize);
            const promises = batch.map(chunk => 
                embeddingModel.embedContent(chunk.content)
                    .then(result => ({
                        embedding: result.embedding.values,
                        content: chunk.content,
                        wordCount: chunk.wordCount
                    }))
            );
            const results = await Promise.all(promises);
            embeddings.push(...results);
        }
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
        return stats.namespaces[namespace]?.recordCount || 0;
    } catch (error) {
        console.error(`Error getting namespace count:`, error);
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

async function processAndUpsert(chunks, fileName, index, embeddingModel) {
    const results = new Map();
    const embeddingsWithMetadata = await generateEmbeddings(chunks, embeddingModel);
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
                await index.namespace(currentNamespaceInfo.namespace).upsert(currentBatch);
                console.log(`Processed batch of ${currentBatch.length} records in namespace ${currentNamespaceInfo.namespace}`);
                currentBatch = [];
            } catch (error) {
                console.error(`Error upserting batch to namespace ${currentNamespaceInfo.namespace}:`, error);
                throw error;
            }
        }
    }

    if (currentBatch.length > 0) {
        try {
            await index.namespace(currentNamespaceInfo.namespace).upsert(currentBatch);
            console.log(`Processed final batch of ${currentBatch.length} records in namespace ${currentNamespaceInfo.namespace}`);
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
        const { pinecone, embeddingModel, index } = await initServices();
        
        // Use custom file path if provided, otherwise default to './data.txt'
        const filePath = customFilePath || './data.txt';
        const fileName = path.basename(filePath);
        
        console.log(`\nProcessing file: ${filePath}`);
        
        const chunks = await readAndProcessFile(filePath);
        const results = await processAndUpsert(chunks, fileName, index, embeddingModel);
        
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