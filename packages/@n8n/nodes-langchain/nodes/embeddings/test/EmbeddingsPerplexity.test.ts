import { aiClientFetch } from '@n8n/ai-utilities';
import { createMockExecuteFunction } from 'n8n-nodes-base/test/nodes/Helpers';
import type { INode, ISupplyDataFunctions } from 'n8n-workflow';
import type { Mocked } from 'vitest';

import { EmbeddingsPerplexity } from '../EmbeddingsPerplexity/EmbeddingsPerplexity.node';
import { PerplexityEmbeddings } from '../EmbeddingsPerplexity/helpers';

vi.mock('../EmbeddingsPerplexity/helpers', () => ({ PerplexityEmbeddings: vi.fn() }));

class MockProxyAgent {}

vi.mock('@n8n/ai-utilities', async () => {
	const actual = await vi.importActual('@n8n/ai-utilities');
	return {
		...actual,
		logWrapper: vi.fn().mockImplementation(() => vi.fn()),
		getProxyAgent: vi.fn().mockImplementation(() => new MockProxyAgent()),
	};
});

const MockedPerplexityEmbeddings = vi.mocked(PerplexityEmbeddings);

describe('EmbeddingsPerplexity', () => {
	let node: EmbeddingsPerplexity;

	const mockNode: INode = {
		id: '1',
		name: 'Embeddings Perplexity',
		typeVersion: 1,
		type: '@n8n/n8n-nodes-langchain.embeddingsPerplexity',
		position: [0, 0],
		parameters: {},
	};

	const setupMockContext = (options: Record<string, unknown> = {}) => {
		const ctx = createMockExecuteFunction<ISupplyDataFunctions>(
			{},
			mockNode,
		) as Mocked<ISupplyDataFunctions>;

		ctx.getCredentials = vi.fn().mockResolvedValue({ apiKey: 'test-key' });
		ctx.getNode = vi.fn().mockReturnValue(mockNode);
		ctx.getNodeParameter = vi.fn().mockImplementation((paramName: string) => {
			if (paramName === 'model') return 'pplx-embed-v1-4b';
			if (paramName === 'options') return options;
			return undefined;
		});
		ctx.logger = {
			debug: vi.fn(),
			info: vi.fn(),
			warn: vi.fn(),
			error: vi.fn(),
		};
		return ctx;
	};

	beforeEach(() => {
		node = new EmbeddingsPerplexity();
		vi.clearAllMocks();
	});

	it('should use the perplexityApi credential and output ai_embedding', () => {
		expect(node.description).toMatchObject({
			name: 'embeddingsPerplexity',
			credentials: [{ name: 'perplexityApi', required: true }],
			outputs: ['ai_embedding'],
		});
	});

	it('should call the Perplexity API with the credential key', async () => {
		const ctx = setupMockContext();

		const result = await node.supplyData.call(ctx, 0);

		expect(ctx.getCredentials).toHaveBeenCalledWith('perplexityApi');
		expect(MockedPerplexityEmbeddings).toHaveBeenCalledWith(
			expect.objectContaining({
				apiKey: 'test-key',
				model: 'pplx-embed-v1-4b',
				configuration: expect.objectContaining({
					baseURL: 'https://api.perplexity.ai/v1',
					defaultHeaders: { 'X-Source': 'n8n' },
					fetch: aiClientFetch,
					fetchOptions: { dispatcher: expect.any(MockProxyAgent) },
				}),
			}),
		);
		expect(result).toHaveProperty('response');
	});

	it('should pass options and treat a timeout of -1 as no timeout', async () => {
		const ctx = setupMockContext({ timeout: -1, batchSize: 128, dimensions: 512 });

		await node.supplyData.call(ctx, 0);

		const callArgs = MockedPerplexityEmbeddings.mock.calls[0][0];
		expect(callArgs).toMatchObject({ batchSize: 128, dimensions: 512 });
		expect(callArgs?.timeout).toBeUndefined();
	});
});
