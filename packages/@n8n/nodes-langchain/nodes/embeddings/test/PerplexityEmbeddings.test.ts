import { decodeBase64Int8, PerplexityEmbeddings } from '../EmbeddingsPerplexity/helpers';

type CreateFn = ReturnType<typeof vi.fn>;

const encode = (values: number[]) => Buffer.from(Int8Array.from(values).buffer).toString('base64');

describe('decodeBase64Int8', () => {
	it('should decode signed int8 values into a unit vector', () => {
		expect(decodeBase64Int8(encode([3, -4]))).toEqual([0.6, -0.8]);
	});

	it('should return a zero vector unchanged', () => {
		expect(decodeBase64Int8(encode([0, 0]))).toEqual([0, 0]);
	});
});

describe('PerplexityEmbeddings', () => {
	const buildEmbeddings = (overrides: Record<string, unknown> = {}) => {
		const create: CreateFn = vi.fn().mockImplementation((body: { input: string[] }) => ({
			data: body.input.map((_, index) => ({ index, embedding: encode([index + 1, 0]) })),
		}));

		const embeddings = new PerplexityEmbeddings({
			apiKey: 'test-key',
			model: 'pplx-embed-v1-0.6b',
			configuration: { baseURL: 'https://api.perplexity.ai/v1' },
			...overrides,
		});

		(embeddings as unknown as { client: { embeddings: { create: CreateFn } } }).client = {
			embeddings: { create },
		};

		return { embeddings, create };
	};

	it('should request base64_int8 and decode each embedding', async () => {
		const { embeddings, create } = buildEmbeddings({ dimensions: 256 });

		const result = await embeddings.embedDocuments(['a', 'b']);

		expect(create.mock.calls[0][0]).toMatchObject({
			model: 'pplx-embed-v1-0.6b',
			input: ['a', 'b'],
			encoding_format: 'base64_int8',
			dimensions: 256,
		});
		expect(result).toEqual([
			[1, 0],
			[1, 0],
		]);
	});

	it('should return a single vector for a query', async () => {
		const { embeddings } = buildEmbeddings();

		expect(await embeddings.embedQuery('what is n8n?')).toEqual([1, 0]);
	});

	it('should order embeddings by index within a batch', async () => {
		const { embeddings, create } = buildEmbeddings();
		create.mockResolvedValueOnce({
			data: [
				{ index: 1, embedding: encode([0, 2]) },
				{ index: 0, embedding: encode([2, 0]) },
			],
		});

		expect(await embeddings.embedDocuments(['first', 'second'])).toEqual([
			[1, 0],
			[0, 1],
		]);
	});

	it('should keep input order across batches', async () => {
		const { embeddings, create } = buildEmbeddings({ batchSize: 1 });
		create
			.mockResolvedValueOnce({ data: [{ index: 0, embedding: encode([5, 0]) }] })
			.mockResolvedValueOnce({ data: [{ index: 0, embedding: encode([0, 5]) }] });

		expect(await embeddings.embedDocuments(['a', 'b'])).toEqual([
			[1, 0],
			[0, 1],
		]);
		expect(create).toHaveBeenCalledTimes(2);
	});

	it('should throw when a batch returns the wrong number of embeddings', async () => {
		const { embeddings, create } = buildEmbeddings();
		create.mockResolvedValueOnce({ data: [{ index: 0, embedding: encode([1]) }] });

		await expect(embeddings.embedDocuments(['a', 'b'])).rejects.toThrow(
			'Perplexity embeddings API returned 1 embeddings for a batch of 2 inputs',
		);
	});
});
