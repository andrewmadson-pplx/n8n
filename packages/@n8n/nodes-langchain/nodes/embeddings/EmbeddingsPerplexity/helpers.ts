import { chunkArray } from '@langchain/core/utils/chunk_array';
import { OpenAIEmbeddings } from '@langchain/openai';
import { OperationalError } from 'n8n-workflow';
import type { OpenAI } from 'openai';

/**
 * Decodes a base64 string of signed 8-bit integers into an L2-normalized vector.
 */
export function decodeBase64Int8(encoded: string): number[] {
	const bytes = Buffer.from(encoded, 'base64');
	const vector = Array.from(new Int8Array(bytes.buffer, bytes.byteOffset, bytes.length));
	const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
	return norm === 0 ? vector : vector.map((value) => value / norm);
}

/**
 * Embeddings for the Perplexity Embeddings API.
 *
 * Perplexity's /v1/embeddings returns base64-encoded int8 vectors and rejects
 * `encoding_format: "float"`. The OpenAI client decodes `base64` as float32, so this class
 * requests `base64_int8` explicitly (the client then returns the raw strings) and decodes
 * each embedding itself.
 */
export class PerplexityEmbeddings extends OpenAIEmbeddings {
	override async embedDocuments(texts: string[]): Promise<number[][]> {
		return await this.embed(texts);
	}

	override async embedQuery(text: string): Promise<number[]> {
		const [embedding] = await this.embed([text]);
		return embedding;
	}

	private async embed(texts: string[]): Promise<number[][]> {
		const input = this.stripNewLines ? texts.map((text) => text.replace(/\n/g, ' ')) : texts;
		const batches = chunkArray(input, this.batchSize);

		const batchResponses = await Promise.all(
			batches.map(async (batch) => {
				const params = {
					model: this.model,
					input: batch,
					encoding_format: 'base64_int8',
					...(this.dimensions ? { dimensions: this.dimensions } : {}),
				} as unknown as OpenAI.EmbeddingCreateParams;
				const { data } = await this.embeddingWithRetry(params);
				return { expected: batch.length, data };
			}),
		);

		const embeddings: number[][] = [];
		for (const { expected, data } of batchResponses) {
			if (data.length !== expected) {
				throw new OperationalError(
					`Perplexity embeddings API returned ${data.length} embeddings for a batch of ${expected} inputs`,
				);
			}
			for (const entry of [...data].sort((a, b) => a.index - b.index)) {
				embeddings.push(decodeBase64Int8(entry.embedding as unknown as string));
			}
		}
		return embeddings;
	}
}
