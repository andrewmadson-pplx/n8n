import { ChatOpenAI, type ClientOptions } from '@langchain/openai';
import {
	getProxyAgent,
	aiClientFetch,
	makeN8nLlmFailedAttemptHandler,
	N8nLlmTracing,
	getConnectionHintNoticeField,
} from '@n8n/ai-utilities';
import {
	NodeConnectionTypes,
	type INodeType,
	type INodeTypeDescription,
	type ISupplyDataFunctions,
	type SupplyData,
} from 'n8n-workflow';

import { MODEL_SELECTION_HINT } from '@utils/model-builder-hints';

import { openAiFailedAttemptHandler } from '../../vendors/OpenAi/helpers/error-handling';

const PERPLEXITY_BASE_URL = 'https://api.perplexity.ai/v1';
// Anthropic models on the Agent API reject requests without max_output_tokens
const DEFAULT_MAX_TOKENS = 4096;

export class LmChatPerplexity implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Perplexity Chat Model',

		name: 'lmChatPerplexity',
		icon: { light: 'file:perplexity.svg', dark: 'file:perplexity.dark.svg' },
		group: ['transform'],
		version: [1],
		description: 'Run models from several providers through the Perplexity Agent API',
		defaults: {
			name: 'Perplexity Chat Model',
		},
		codex: {
			categories: ['AI'],
			subcategories: {
				AI: ['Language Models', 'Root Nodes'],
				'Language Models': ['Chat Models (Recommended)'],
			},
			resources: {
				primaryDocumentation: [
					{
						url: 'https://docs.n8n.io/integrations/builtin/cluster-nodes/sub-nodes/n8n-nodes-langchain.lmchatperplexity/',
					},
				],
			},
			alias: ['perplexity', 'pplx', 'sonar'],
		},

		inputs: [],

		outputs: [NodeConnectionTypes.AiLanguageModel],
		outputNames: ['Model'],
		credentials: [
			{
				name: 'perplexityApi',
				required: true,
			},
		],
		requestDefaults: {
			ignoreHttpStatusErrors: true,
			baseURL: PERPLEXITY_BASE_URL,
		},
		properties: [
			getConnectionHintNoticeField([NodeConnectionTypes.AiChain, NodeConnectionTypes.AiAgent]),
			{
				displayName: 'Model',
				name: 'model',
				type: 'options',
				description:
					'The model which will generate the completion. <a href="https://docs.perplexity.ai/docs/agent-api/models">Learn more</a>.',
				typeOptions: {
					loadOptions: {
						routing: {
							request: {
								method: 'GET',
								url: '/models',
							},
							output: {
								postReceive: [
									{
										type: 'rootProperty',
										properties: {
											property: 'data',
										},
									},
									{
										type: 'setKeyValue',
										properties: {
											name: '={{$responseItem.id}}',
											value: '={{$responseItem.id}}',
										},
									},
									{
										type: 'sort',
										properties: {
											key: 'name',
										},
									},
								],
							},
						},
					},
				},
				routing: {
					send: {
						type: 'body',
						property: 'model',
					},
				},
				default: 'perplexity/sonar',
				builderHint: {
					propertyHint:
						"Choose a model from the connected credential's model list. " + MODEL_SELECTION_HINT,
				},
			},
			{
				displayName: 'Built-in Tools',
				name: 'builtInTools',
				placeholder: 'Add Built-in Tool',
				type: 'collection',
				default: {},
				options: [
					{
						displayName: 'Web Search',
						name: 'webSearch',
						type: 'boolean',
						default: true,
						description: 'Whether to let the model search the web',
					},
					{
						displayName: 'Fetch URL',
						name: 'fetchUrl',
						type: 'boolean',
						default: true,
						description: 'Whether to let the model read the content of a URL',
					},
				],
			},
			{
				displayName: 'Options',
				name: 'options',
				placeholder: 'Add Option',
				description: 'Additional options to add',
				type: 'collection',
				default: {},
				options: [
					{
						displayName: 'Maximum Number of Tokens',
						name: 'maxTokens',
						default: DEFAULT_MAX_TOKENS,
						description: 'The maximum number of tokens to generate in the response',
						type: 'number',
					},
					{
						displayName: 'Sampling Temperature',
						name: 'temperature',
						default: 0.7,
						typeOptions: { maxValue: 2, minValue: 0, numberPrecision: 1 },
						description:
							'Controls randomness: Lowering results in less random completions. As the temperature approaches zero, the model will become deterministic and repetitive.',
						type: 'number',
					},
					{
						displayName: 'Timeout',
						name: 'timeout',
						default: 360000,
						description: 'Maximum amount of time a request is allowed to take in milliseconds',
						type: 'number',
					},
					{
						displayName: 'Max Retries',
						name: 'maxRetries',
						default: 2,
						description: 'Maximum number of retries to attempt',
						type: 'number',
					},
					{
						displayName: 'Top P',
						name: 'topP',
						default: 1,
						typeOptions: { maxValue: 1, minValue: 0, numberPrecision: 1 },
						description:
							'Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered. We generally recommend altering this or temperature but not both.',
						type: 'number',
					},
				],
			},
		],
	};

	async supplyData(this: ISupplyDataFunctions, itemIndex: number): Promise<SupplyData> {
		const credentials = await this.getCredentials<{ apiKey: string }>('perplexityApi');

		const modelName = this.getNodeParameter('model', itemIndex) as string;

		const builtInTools = this.getNodeParameter('builtInTools', itemIndex, {}) as {
			webSearch?: boolean;
			fetchUrl?: boolean;
		};

		const options = this.getNodeParameter('options', itemIndex, {}) as {
			maxTokens?: number;
			maxRetries: number;
			timeout: number;
			temperature?: number;
			topP?: number;
		};

		const timeout = options.timeout;
		const configuration: ClientOptions = {
			baseURL: PERPLEXITY_BASE_URL,
			defaultHeaders: { 'X-Source': 'n8n' },
			fetch: aiClientFetch,
			fetchOptions: {
				dispatcher: getProxyAgent(
					PERPLEXITY_BASE_URL,
					{
						headersTimeout: timeout,
						bodyTimeout: timeout,
					},
					this.helpers.getSecureEgressFilter(),
				),
			},
		};

		const model = new ChatOpenAI({
			apiKey: credentials.apiKey,
			model: modelName,
			...options,
			maxTokens: options.maxTokens ?? DEFAULT_MAX_TOKENS,
			timeout,
			maxRetries: options.maxRetries ?? 2,
			configuration,
			callbacks: [new N8nLlmTracing(this)],
			onFailedAttempt: makeN8nLlmFailedAttemptHandler(this, openAiFailedAttemptHandler),
			useResponsesApi: true,
			supportsStrictToolCalling: false,
		});

		const tools = [];
		if (builtInTools.webSearch) tools.push({ type: 'web_search' });
		if (builtInTools.fetchUrl) tools.push({ type: 'fetch_url' });
		// the Tools Agent reads built-in tools from the model metadata
		if (tools.length) {
			model.metadata = { ...model.metadata, tools };
		}

		return {
			response: model,
		};
	}
}
