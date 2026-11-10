import { DynamicTool } from '@langchain/core/tools';
import type {
	IExecuteFunctions,
	INode,
	INodeExecutionData,
	ISupplyDataFunctions,
} from 'n8n-workflow';
import { mock } from 'vitest-mock-extended';

import { ToolPerplexitySearch } from './ToolPerplexitySearch.node';

const PERPLEXITY_RESPONSE = {
	id: 'test-id',
	results: [
		{
			title: 'First result',
			url: 'https://example.com/1',
			snippet: 'First snippet',
			date: '2026-01-01',
			last_updated: '2026-01-02',
		},
		{
			title: 'Second result',
			url: 'https://example.com/2',
			snippet: 'Second snippet',
		},
	],
};

const EXPECTED_RESPONSE = JSON.stringify([
	{
		title: 'First result',
		url: 'https://example.com/1',
		snippet: 'First snippet',
		date: '2026-01-01',
	},
	{
		title: 'Second result',
		url: 'https://example.com/2',
		snippet: 'Second snippet',
		date: null,
	},
]);

function createContext(
	overrides: {
		options?: Record<string, unknown>;
		httpResponse?: unknown;
		inputData?: INodeExecutionData[];
	} = {},
) {
	const httpRequestWithAuthentication = vi
		.fn()
		.mockResolvedValue(overrides.httpResponse ?? PERPLEXITY_RESPONSE);
	const ctx = mock<IExecuteFunctions>({
		getInputData: vi.fn(() => overrides.inputData ?? [{ json: { input: 'test' } }]),
		getNode: vi.fn(() => mock<INode>({ name: 'test perplexity search' })),
		getNodeParameter: vi.fn().mockReturnValue(overrides.options ?? {}),
		addInputData: vi.fn().mockReturnValue({ index: 0 }),
		addOutputData: vi.fn(),
	});
	ctx.helpers.httpRequestWithAuthentication = httpRequestWithAuthentication;
	return { ctx, httpRequestWithAuthentication };
}

describe('ToolPerplexitySearch', () => {
	beforeEach(() => {
		vi.resetAllMocks();
	});

	describe('supplyData', () => {
		it('should return a search tool that calls the Search API with the Perplexity credential', async () => {
			const node = new ToolPerplexitySearch();
			const { ctx, httpRequestWithAuthentication } = createContext();

			const supplyDataResult = await node.supplyData.call(
				ctx as unknown as ISupplyDataFunctions,
				0,
			);

			const tool = supplyDataResult.response as DynamicTool;
			expect(tool).toBeInstanceOf(DynamicTool);
			expect(tool.name).toBe('perplexity-search');

			const response = await tool.invoke('current events');

			expect(httpRequestWithAuthentication).toHaveBeenCalledWith('perplexityApi', {
				method: 'POST',
				url: 'https://api.perplexity.ai/search',
				body: { query: 'current events', max_results: 10 },
				json: true,
				timeout: 30_000,
			});
			expect(response).toBe(EXPECTED_RESPONSE);
		});

		it('should apply configured options to the request body', async () => {
			const node = new ToolPerplexitySearch();
			const { ctx, httpRequestWithAuthentication } = createContext({
				options: {
					maxResults: 3,
					country: 'US',
					searchRecencyFilter: 'week',
					searchDomainFilter: 'nature.com, science.org,',
					searchLanguageFilter: 'en,fr',
				},
			});

			const supplyDataResult = await node.supplyData.call(
				ctx as unknown as ISupplyDataFunctions,
				0,
			);
			await (supplyDataResult.response as DynamicTool).invoke('test');

			expect(httpRequestWithAuthentication).toHaveBeenCalledWith(
				'perplexityApi',
				expect.objectContaining({
					body: {
						query: 'test',
						max_results: 3,
						country: 'US',
						search_recency_filter: 'week',
						search_domain_filter: ['nature.com', 'science.org'],
						search_language_filter: ['en', 'fr'],
					},
				}),
			);
		});

		it('should report when there are no results', async () => {
			const node = new ToolPerplexitySearch();
			const { ctx } = createContext({ httpResponse: { id: 'test-id', results: [] } });

			const supplyDataResult = await node.supplyData.call(
				ctx as unknown as ISupplyDataFunctions,
				0,
			);
			const response = await (supplyDataResult.response as DynamicTool).invoke('test');

			expect(response).toBe('No good results found.');
		});
	});

	describe('execute', () => {
		it('should execute a search per input item and return results', async () => {
			const node = new ToolPerplexitySearch();
			const { ctx, httpRequestWithAuthentication } = createContext({
				inputData: [{ json: { input: 'machine learning' } }, { json: { input: 'deep learning' } }],
			});

			const result = await node.execute.call(ctx);

			expect(httpRequestWithAuthentication).toHaveBeenCalledTimes(2);
			expect(httpRequestWithAuthentication).toHaveBeenNthCalledWith(
				1,
				'perplexityApi',
				expect.objectContaining({ body: expect.objectContaining({ query: 'machine learning' }) }),
			);
			expect(httpRequestWithAuthentication).toHaveBeenNthCalledWith(
				2,
				'perplexityApi',
				expect.objectContaining({ body: expect.objectContaining({ query: 'deep learning' }) }),
			);
			expect(result).toEqual([
				[
					{ json: { response: EXPECTED_RESPONSE }, pairedItem: { item: 0 } },
					{ json: { response: EXPECTED_RESPONSE }, pairedItem: { item: 1 } },
				],
			]);
		});

		it('should throw when an input item has no query', async () => {
			const node = new ToolPerplexitySearch();
			const { ctx, httpRequestWithAuthentication } = createContext({ inputData: [{ json: {} }] });

			await expect(node.execute.call(ctx)).rejects.toThrow('Input item is missing');
			expect(httpRequestWithAuthentication).not.toHaveBeenCalled();
		});

		it('should read options per item', async () => {
			const node = new ToolPerplexitySearch();
			const { ctx } = createContext({ inputData: [{ json: { input: 'test query' } }] });

			await node.execute.call(ctx);

			expect(ctx.getNodeParameter).toHaveBeenCalledWith('options', 0);
		});
	});
});
