// rag_uploader.js
import { Pinecone } from '@pinecone-database/pinecone';
import { createMistral } from '@ai-sdk/mistral';
import { embedMany } from 'ai';
import { encode as gptEncode } from 'gpt-tokenizer';
import fs from 'fs/promises';
import path from 'path';

const config = {
  indexName: 'rag',
  dimension: 1024, // Mistral embedding dimension
  batchSize: 10,
  recordsPerNamespace: 300,
  wordsPerChunk: 350,
  maxTokensPerChunk: 350,
  overlapTokens: 80,   // increased to reduce clause-splitting errors
  maxWordsPerChunk: 350,
  overlapWords: 50
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

/* ---------------------------
   Token & text utilities
   --------------------------- */

const encoding = { encode: (t) => gptEncode(t) };

function countTokens(text) {
  if (!text) return 0;
  try {
    if (encoding) return encoding.encode(text).length;
  } catch (e) {
    // fallback to word count
  }
  return text.split(/\s+/).filter(Boolean).length;
}

function splitMarkdownIntoSections(markdownText) {
  const lines = markdownText.split(/\n/);
  const sections = [];
  let current = { heading: 'Document', depth: 0, contentLines: [] };
  const headingRegex = /^#{1,6}\s+(.*)$/;

  for (const line of lines) {
    const match = line.match(headingRegex);
    if (match) {
      if (current.contentLines.length > 0) {
        sections.push({
          heading: current.heading,
          depth: current.depth,
          content: current.contentLines.join('\n').trim()
        });
      }
      const depth = (line.match(/^#+/)[0] || '').length;
      current = { heading: match[1].trim(), depth, contentLines: [] };
    } else {
      current.contentLines.push(line);
    }
  }
  if (current.contentLines.length > 0) {
    sections.push({
      heading: current.heading,
      depth: current.depth,
      content: current.contentLines.join('\n').trim()
    });
  }
  // fallback: if no headings detected, return [] (caller will handle)
  return sections.filter(s => s.content && s.content.trim().length > 0);
}

function groupLinesIntoBlocksPreservingLists(sectionContent) {
  const lines = sectionContent.split(/\n/);
  const blocks = [];
  let currentBlock = [];
  let currentIsList = false;
  const isListLine = (l) => /^(\s*)([-*+]|•|◦|\d+[.)])\s+/.test(l);
  const isIndented = (l) => /^\s{2,}\S/.test(l);

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const listy = isListLine(line);
    if (currentBlock.length === 0) {
      currentIsList = listy;
      currentBlock.push(line);
      continue;
    }
    if (currentIsList) {
      if (listy || isIndented(line) || line.trim() === '') {
        currentBlock.push(line);
      } else {
        blocks.push(currentBlock.join('\n').trim());
        currentBlock = [line];
        currentIsList = isListLine(line);
      }
    } else {
      if (line.trim() === '') {
        currentBlock.push(line);
        blocks.push(currentBlock.join('\n').trim());
        currentBlock = [];
        currentIsList = false;
      } else if (listy) {
        if (currentBlock.length) {
          blocks.push(currentBlock.join('\n').trim());
        }
        currentBlock = [line];
        currentIsList = true;
      } else {
        currentBlock.push(line);
      }
    }
  }
  if (currentBlock.length) blocks.push(currentBlock.join('\n').trim());
  return blocks.filter(b => b && b.replace(/\n/g, '').trim().length > 0);
}

function chunkSectionByTokens(sectionContent, heading, depth, maxTokens, overlapTokens) {
  const blocks = groupLinesIntoBlocksPreservingLists(sectionContent);
  const blockTokenCounts = blocks.map(b => countTokens(b));
  const chunks = [];
  let currentBlocks = [];
  let currentTokens = 0;

  const flushChunk = () => {
    if (currentBlocks.length === 0) return;
    const content = currentBlocks.join('\n\n');
    chunks.push({
      content,
      wordCount: content.split(/\s+/).filter(Boolean).length,
      sectionHeading: heading,
      sectionDepth: depth
    });
  };

  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i];
    const tokens = blockTokenCounts[i];
    if (currentTokens + tokens > maxTokens && currentBlocks.length > 0) {
      flushChunk();
      // create overlap using last few blocks whose combined tokens <= overlapTokens
      let overlapBlocks = [];
      let overlapSum = 0;
      for (let j = currentBlocks.length - 1; j >= 0; j--) {
        const bt = countTokens(currentBlocks[j]);
        if (overlapSum + bt > overlapTokens) break;
        overlapBlocks.unshift(currentBlocks[j]);
        overlapSum += bt;
      }
      currentBlocks = overlapBlocks.slice();
      currentTokens = overlapSum;
    }
    currentBlocks.push(block);
    currentTokens += tokens;
  }
  flushChunk();
  return chunks;
}

/* ---------------------------
   File read & chunking (patched)
   --------------------------- */

async function readAndProcessFile(filePath) {
  try {
    const content = await fs.readFile(filePath, 'utf8');

    if (!content || content.trim().length === 0) {
      throw new Error('File is empty or contains only whitespace');
    }

    // Split by markdown-style headings first. If none found, treat whole doc as single section.
    let sections = splitMarkdownIntoSections(content);
    if (!sections || sections.length === 0) {
      sections = [{ heading: 'Document', depth: 0, content }];
    }

    const allChunks = [];

    for (let sIdx = 0; sIdx < sections.length; sIdx++) {
      const s = sections[sIdx];
      const sectionChunks = chunkSectionByTokens(
        s.content,
        s.heading,
        s.depth,
        config.maxTokensPerChunk,
        config.overlapTokens
      );

      sectionChunks.forEach((chunk, cIdx) => {
        allChunks.push({
          content: chunk.content,
          wordCount: chunk.wordCount,
          sectionHeading: chunk.sectionHeading || s.heading,
          sectionDepth: chunk.sectionDepth || s.depth,
          sectionIndex: sIdx + 1,
          sectionChunkIndex: cIdx + 1
        });
      });
    }

    // Merge very small chunks (<50 words) into nearest neighbor to avoid low-signal fragments
    const merged = [];
    for (let i = 0; i < allChunks.length; i++) {
      const cur = allChunks[i];
      if (cur.wordCount < 50 && merged.length > 0) {
        const prev = merged[merged.length - 1];
        if (prev.sectionIndex === cur.sectionIndex) {
          prev.content = prev.content + '\n\n' + cur.content;
          prev.wordCount += cur.wordCount;
        } else if (i + 1 < allChunks.length) {
          allChunks[i + 1].content = cur.content + '\n\n' + allChunks[i + 1].content;
          allChunks[i + 1].wordCount += cur.wordCount;
        } else {
          prev.content = prev.content + '\n\n' + cur.content;
          prev.wordCount += cur.wordCount;
        }
      } else {
        merged.push(cur);
      }
    }

    const finalChunks = merged
      .map((c, idx) => ({ ...c, content: c.content.trim(), finalIndex: idx + 1 }))
      .filter(c => c.content && c.content.length > 20);

    console.log(`Read file ${filePath}: sections=${sections.length}, initialChunks=${allChunks.length}, finalChunks=${finalChunks.length}`);
    return finalChunks;
  } catch (error) {
    console.error(`Error processing file ${filePath}:`, error);
    throw error;
  }
}

/* ---------------------------
   Embedding generation (unchanged)
   --------------------------- */

async function generateEmbeddings(chunks) {
  try {
    console.log(`Generating embeddings for ${chunks.length} chunks`);
    const embeddings = [];
    const batchSize = Math.max(1, config.batchSize);

    for (let i = 0; i < chunks.length; i += batchSize) {
      const batch = chunks.slice(i, i + batchSize);
      console.log(`Processing batch ${Math.floor(i / batchSize) + 1}, chunks ${i + 1}-${Math.min(i + batchSize, chunks.length)}`);

      const texts = batch.map(chunk => chunk.content);

      const { embeddings: batchEmbeddings } = await embedMany({
        model: mistral.textEmbeddingModel('mistral-embed'),
        values: texts,
      });

      const results = batchEmbeddings.map((embedding, index) => ({
        embedding: embedding,
        content: batch[index].content,
        wordCount: batch[index].wordCount,
        sectionHeading: batch[index].sectionHeading,
        sectionDepth: batch[index].sectionDepth,
        sectionIndex: batch[index].sectionIndex,
        sectionChunkIndex: batch[index].sectionChunkIndex
      }));

      console.log(`Generated embeddings for batch ${Math.floor(i / batchSize) + 1}, embedding length: ${results[0].embedding.length}`);
      embeddings.push(...results);
    }
    console.log(`Total embeddings generated: ${embeddings.length}`);
    return embeddings;
  } catch (error) {
    console.error("Error in batch embedding generation:", error);
    throw error;
  }
}

/* ---------------------------
   Namespace helpers (unchanged)
   --------------------------- */

async function getCurrentNamespaceCount(index, namespace) {
  try {
    const stats = await index.describeIndexStats({
      filter: { namespace: namespace }
    });
    console.log(`Namespace ${namespace} stats:`, stats.namespaces?.[namespace]);
    return stats.namespaces?.[namespace]?.recordCount || 0;
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

/* ---------------------------
   Reliable upsert + processing
   --------------------------- */

async function safeUpsertToNamespace(index, namespace, records, maxRetries = 2) {
  let attempt = 0;
  while (attempt <= maxRetries) {
    try {
      const res = await index.namespace(namespace).upsert(records);
      console.log(`Upsert to ${namespace} succeeded (attempt ${attempt + 1}).`, res?.upsertedCount ?? 'no-response-field');
      return res;
    } catch (err) {
      attempt++;
      console.error(`Upsert to ${namespace} failed on attempt ${attempt}:`, err?.message || err);
      if (attempt > maxRetries) throw err;
      await new Promise(r => setTimeout(r, 500 * attempt));
    }
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
        await safeUpsertToNamespace(index, currentNamespaceInfo.namespace, currentBatch);
      }
      currentNamespaceInfo = await getNextNamespace(index);
      currentBatch = [];
      console.log(`Switching to namespace: ${currentNamespaceInfo.namespace}`);
    }

    const record = {
      id: `${fileName.replace(/[^a-zA-Z0-9]/g, '_')}_s${embedData.sectionIndex ?? 0}_c${embedData.sectionChunkIndex ?? processedCount}`,
      values: embedData.embedding,
      metadata: {
        fileName,
        content: embedData.content,
        wordCount: embedData.wordCount,
        sectionHeading: embedData.sectionHeading ?? null,
        sectionDepth: embedData.sectionDepth ?? null,
        sectionIndex: embedData.sectionIndex ?? null,
        sectionChunkIndex: embedData.sectionChunkIndex ?? null,
        namespace: currentNamespaceInfo.namespace,
        timestamp: new Date().toISOString()
      }
    };

    currentBatch.push(record);
    currentNamespaceInfo.currentCount++;
    processedCount++;

    if (!results.has(currentNamespaceInfo.namespace)) results.set(currentNamespaceInfo.namespace, 0);
    results.set(currentNamespaceInfo.namespace, results.get(currentNamespaceInfo.namespace) + 1);

    if (currentBatch.length >= config.batchSize) {
      console.log(`Upserting batch of ${currentBatch.length} records to namespace ${currentNamespaceInfo.namespace}`);
      await safeUpsertToNamespace(index, currentNamespaceInfo.namespace, currentBatch);
      currentBatch = [];
    }
  }

  if (currentBatch.length > 0) {
    console.log(`Upserting final batch of ${currentBatch.length} records to namespace ${currentNamespaceInfo.namespace}`);
    await safeUpsertToNamespace(index, currentNamespaceInfo.namespace, currentBatch);
  }

  return results;
}

/* ---------------------------
   Verification helper
   --------------------------- */

async function verifyIndexForQuery(index, queryText, topK = 5) {
  try {
    const { embeddings } = await embedMany({
      model: mistral.textEmbeddingModel('mistral-embed'),
      values: [queryText]
    });
    const qvec = embeddings[0];

    const queryRes = await index.query({
      vector: qvec,
      topK,
      includeMetadata: true
    });

    const matches = (queryRes.matches || queryRes).slice(0, topK).map(m => ({
      id: m.id || m._id || m.name,
      score: m.score ?? m.similarity ?? m[ 'score' ],
      metadataPreview: m.metadata ? (m.metadata.content ? (m.metadata.content.slice(0, 400) + (m.metadata.content.length > 400 ? '...' : '')) : m.metadata) : null
    }));

    console.log(`Query results for "${queryText}":`, matches);
    return queryRes;
  } catch (err) {
    console.error('verifyIndexForQuery error:', err);
    throw err;
  }
}

/* ---------------------------
   Main flow
   --------------------------- */

async function main(customFilePath = null) {
  try {
    const { pinecone, index } = await initServices();

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

    // Verification example (uncomment if you want auto-check)
    // await verifyIndexForQuery(index, 'organ donor', 10);

    return {
      processedChunks: chunks.length,
      namespaces: resultObj,
      totalVectors: Object.values(resultObj).reduce((sum, count) => sum + count, 0),
      indexStats: stats
    };

  } catch (error) {
    console.error('Error in main process:', error);
    throw error;
  }
}

process.on('unhandledRejection', (error) => {
  console.error('Unhandled promise rejection:', error);
  process.exit(1);
});

// Run when executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().then(res => {
    console.log('Done:', res);
  }).catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
  });
}

export {
  initServices,
  readAndProcessFile,
  processAndUpsert,
  main,
  verifyIndexForQuery
};
