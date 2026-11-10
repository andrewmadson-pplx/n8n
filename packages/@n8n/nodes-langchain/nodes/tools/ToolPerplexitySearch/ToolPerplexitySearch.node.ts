import { DynamicTool } from '@langchain/core/tools';
import { logWrapper, getConnectionHintNoticeField } from '@n8n/ai-utilities';
import { NodeConnectionTypes, NodeOperationError } from 'n8n-workflow';
import type {
	IDataObject,
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	ISupplyDataFunctions,
	SupplyData,
} from 'n8n-workflow';

type Options = {
	maxResults?: number;
	country?: string;
	searchRecencyFilter?: string;
	searchDomainFilter?: string;
	searchLanguageFilter?: string;
};

type PerplexitySearchResponse = {
	results?: Array<{ title?: string; url?: string; snippet?: string; date?: string | null }>;
};

function splitList(value: string): string[] {
	return value
		.split(',')
		.map((entry) => entry.trim())
		.filter(Boolean);
}

function formatResponse(data: PerplexitySearchResponse): string {
	const results = data.results ?? [];
	if (!results.length) return 'No good results found.';

	return JSON.stringify(
		results.map((result) => ({
			title: result.title ?? '',
			url: result.url ?? '',
			snippet: result.snippet ?? '',
			date: result.date ?? null,
		})),
	);
}

function getTool(ctx: ISupplyDataFunctions | IExecuteFunctions, itemIndex: number) {
	const options = ctx.getNodeParameter('options', itemIndex) as Options;

	const body: IDataObject = { max_results: options.maxResults ?? 10 };
	if (options.country) body.country = options.country;
	if (options.searchRecencyFilter) body.search_recency_filter = options.searchRecencyFilter;
	if (options.searchDomainFilter) body.search_domain_filter = splitList(options.searchDomainFilter);
	if (options.searchLanguageFilter) {
		body.search_language_filter = splitList(options.searchLanguageFilter);
	}

	return new DynamicTool({
		name: 'perplexity-search',
		description:
			'A web search engine. Useful for when you need to answer questions about current events or find up-to-date information. Input should be a search query. Output is a JSON array of ranked results with title, url, snippet and date.',
		func: async (query: string) => {
			const response = (await ctx.helpers.httpRequestWithAuthentication.call(ctx, 'perplexityApi', {
				method: 'POST',
				url: 'https://api.perplexity.ai/search',
				body: { query, ...body },
				json: true,
				timeout: 30_000,
			})) as PerplexitySearchResponse;

			return formatResponse(response);
		},
	});
}

export class ToolPerplexitySearch implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Perplexity Search',
		name: 'toolPerplexitySearch',
		icon: { light: 'file:perplexity.svg', dark: 'file:perplexity.dark.svg' },
		group: ['transform'],
		version: 1,
		description: 'Search the web using the Perplexity Search API',
		defaults: {
			name: 'Perplexity Search',
		},
		codex: {
			categories: ['AI'],
			subcategories: {
				AI: ['Tools'],
				Tools: ['Other Tools'],
			},
			resources: {
				primaryDocumentation: [
					{
						url: 'https://docs.n8n.io/integrations/builtin/cluster-nodes/sub-nodes/n8n-nodes-langchain.toolperplexitysearch/',
					},
				],
			},
		},
		inputs: [],
		outputs: [NodeConnectionTypes.AiTool],
		outputNames: ['Tool'],
		credentials: [
			{
				name: 'perplexityApi',
				required: true,
			},
		],
		properties: [
			getConnectionHintNoticeField([NodeConnectionTypes.AiAgent]),
			{
				displayName: 'Options',
				name: 'options',
				type: 'collection',
				placeholder: 'Add Option',
				default: {},
				options: [
					{
						displayName: 'Country',
						name: 'country',
						type: 'string',
						default: '',
						placeholder: 'e.g. US',
						description: 'ISO 3166-1 alpha-2 country code to bias results toward',
					},
					{
						displayName: 'Domain Filter',
						name: 'searchDomainFilter',
						type: 'string',
						default: '',
						placeholder: 'e.g. nature.com, science.org',
						description:
							'Comma-separated list of up to 20 domains to include. To exclude domains instead, prefix every domain with "-". Include and exclude entries cannot be mixed.',
					},
					{
						displayName: 'Language Filter',
						name: 'searchLanguageFilter',
						type: 'string',
						default: '',
						placeholder: 'e.g. en, fr',
						description: 'Comma-separated list of ISO 639-1 language codes',
					},
					{
						displayName: 'Max Results',
						name: 'maxResults',
						type: 'number',
						typeOptions: { minValue: 1, maxValue: 20 },
						default: 10,
						description: 'Max number of results to return',
					},
					{
						displayName: 'Recency',
						name: 'searchRecencyFilter',
						type: 'options',
						options: [
							{ name: 'Any Time', value: '' },
							{ name: 'Past Day', value: 'day' },
							{ name: 'Past Hour', value: 'hour' },
							{ name: 'Past Month', value: 'month' },
							{ name: 'Past Week', value: 'week' },
							{ name: 'Past Year', value: 'year' },
						],
						default: '',
						description: 'Only return results published within this period',
					},
				],
			},
		],
	};

	async supplyData(this: ISupplyDataFunctions, itemIndex: number): Promise<SupplyData> {
		return {
			response: logWrapper(getTool(this, itemIndex), this),
		};
	}

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const result: INodeExecutionData[] = [];
		const input = this.getInputData();
		for (let i = 0; i < input.length; i++) {
			const item = input[i];
			const tool = getTool(this, i);

			const query = item.json.input;
			if (typeof query !== 'string' || query === '') {
				throw new NodeOperationError(this.getNode(), 'Input item is missing', {
					itemIndex: i,
				});
			}

			result.push({
				json: {
					response: await tool.invoke(query),
				},
				pairedItem: {
					item: i,
				},
			});
		}

		return [result];
	}
}
