import { ChatOpenAI } from '@langchain/openai';
import { makeN8nLlmFailedAttemptHandler, getProxyAgent, aiClientFetch } from '@n8n/ai-utilities';
import { createMockExecuteFunction } from 'n8n-nodes-base/test/nodes/Helpers';
import type { INode, ISupplyDataFunctions } from 'n8n-workflow';
import type { Mocked } from 'vitest';

import { LmChatPerplexity } from '../LmChatPerplexity.node';

vi.mock('@langchain/openai');
vi.mock('@n8n/ai-utilities');

const MockedChatOpenAI = vi.mocked(ChatOpenAI);
const mockedMakeN8nLlmFailedAttemptHandler = vi.mocked(makeN8nLlmFailedAttemptHandler);
const mockedGetProxyAgent = vi.mocked(getProxyAgent);
const mockedAiClientFetch = vi.mocked(aiClientFetch);

describe('LmChatPerplexity', () => {
	let node: LmChatPerplexity;

	const mockNodeDef: INode = {
		id: '1',
		name: 'Perplexity Chat Model',
		typeVersion: 1,
		type: '@n8n/n8n-nodes-langchain.lmChatPerplexity',
		position: [0, 0],
		parameters: {},
	};

	const setupMockContext = (params: Record<string, unknown> = {}) => {
		const ctx = createMockExecuteFunction<ISupplyDataFunctions>(
			{},
			mockNodeDef,
		) as Mocked<ISupplyDataFunctions>;

		ctx.getCredentials = vi.fn().mockResolvedValue({ apiKey: 'test-perplexity-key' });
		ctx.getNode = vi.fn().mockReturnValue(mockNodeDef);
		ctx.getNodeParameter = vi.fn().mockImplementation((paramName: string) => {
			if (paramName in params) return params[paramName];
			if (paramName === 'model') return 'perplexity/sonar';
			if (paramName === 'builtInTools' || paramName === 'options') return {};
			return undefined;
		});

		mockedMakeN8nLlmFailedAttemptHandler.mockReturnValue(vi.fn());
		mockedGetProxyAgent.mockReturnValue({} as any);
		return ctx;
	};

	beforeEach(() => {
		node = new LmChatPerplexity();
		vi.clearAllMocks();
	});

	describe('node description', () => {
		it('should have correct node properties', () => {
			expect(node.description).toMatchObject({
				displayName: 'Perplexity Chat Model',
				name: 'lmChatPerplexity',
				group: ['transform'],
				version: [1],
			});
		});

		it('should require perplexityApi credentials', () => {
			expect(node.description.credentials).toEqual([{ name: 'perplexityApi', required: true }]);
		});

		it('should output ai_languageModel', () => {
			expect(node.description.outputs).toEqual(['ai_languageModel']);
			expect(node.description.outputNames).toEqual(['Model']);
		});
	});

	describe('supplyData', () => {
		it('should create ChatOpenAI on the Agent API', async () => {
			const ctx = setupMockContext();

			const result = await node.supplyData.call(ctx, 0);

			expect(ctx.getCredentials).toHaveBeenCalledWith('perplexityApi');
			expect(MockedChatOpenAI).toHaveBeenCalledWith(
				expect.objectContaining({
					apiKey: 'test-perplexity-key',
					model: 'perplexity/sonar',
					maxRetries: 2,
					maxTokens: 4096,
					useResponsesApi: true,
					supportsStrictToolCalling: false,
					callbacks: expect.arrayContaining([expect.any(Object)]),
					onFailedAttempt: expect.any(Function),
					configuration: expect.objectContaining({
						fetch: mockedAiClientFetch,
						baseURL: 'https://api.perplexity.ai/v1',
						defaultHeaders: { 'X-Source': 'n8n' },
					}),
				}),
			);
			expect(result).toEqual({ response: expect.any(Object) });
			expect((result.response as ChatOpenAI).metadata).toBeUndefined();
		});

		it('should pass options to ChatOpenAI', async () => {
			const ctx = setupMockContext({
				model: 'anthropic/claude-sonnet-4-6',
				options: { temperature: 0.5, maxTokens: 2000, topP: 0.9, timeout: 60000, maxRetries: 5 },
			});

			await node.supplyData.call(ctx, 0);

			expect(MockedChatOpenAI).toHaveBeenCalledWith(
				expect.objectContaining({
					model: 'anthropic/claude-sonnet-4-6',
					temperature: 0.5,
					maxTokens: 2000,
					topP: 0.9,
					timeout: 60000,
					maxRetries: 5,
				}),
			);
			expect(mockedGetProxyAgent).toHaveBeenCalledWith(
				'https://api.perplexity.ai/v1',
				expect.objectContaining({ headersTimeout: 60000, bodyTimeout: 60000 }),
				ctx.helpers.getSecureEgressFilter(),
			);
		});

		it('should add enabled built-in tools to the model metadata', async () => {
			const ctx = setupMockContext({ builtInTools: { webSearch: true, fetchUrl: true } });

			const result = await node.supplyData.call(ctx, 0);

			expect((result.response as ChatOpenAI).metadata).toEqual({
				tools: [{ type: 'web_search' }, { type: 'fetch_url' }],
			});
		});
	});
});
