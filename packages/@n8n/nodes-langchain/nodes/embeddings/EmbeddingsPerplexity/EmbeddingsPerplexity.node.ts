import type { ClientOptions } from '@langchain/openai';
import {
	getProxyAgent,
	aiClientFetch,
	logWrapper,
	getConnectionHintNoticeField,
} from '@n8n/ai-utilities';
import {
	NodeConnectionTypes,
	type INodeType,
	type INodeTypeDescription,
	type ISupplyDataFunctions,
	type SupplyData,
} from 'n8n-workflow';

import { PerplexityEmbeddings } from './helpers';

const PERPLEXITY_BASE_URL = 'https://api.perplexity.ai/v1';

export class EmbeddingsPerplexity implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Embeddings Perplexity',
		name: 'embeddingsPerplexity',
		icon: { light: 'file:perplexity.svg', dark: 'file:perplexity.dark.svg' },
		group: ['transform'],
		version: [1],
		description: 'Use Perplexity Embeddings',
		defaults: {
			name: 'Embeddings Perplexity',
		},
		credentials: [
			{
				name: 'perplexityApi',
				required: true,
			},
		],
		codex: {
			categories: ['AI'],
			subcategories: {
				AI: ['Embeddings'],
			},
			alias: ['perplexity', 'pplx', 'embeddings'],
			resources: {
				primaryDocumentation: [
					{
						url: 'https://docs.n8n.io/integrations/builtin/cluster-nodes/sub-nodes/n8n-nodes-langchain.embeddingsperplexity/',
					},
				],
			},
		},
		inputs: [],
		outputs: [NodeConnectionTypes.AiEmbedding],
		outputNames: ['Embeddings'],
		properties: [
			getConnectionHintNoticeField([NodeConnectionTypes.AiVectorStore]),
			{
				displayName: 'Model',
				name: 'model',
				type: 'options',
				default: 'pplx-embed-v1-0.6b',
				options: [
					{
						name: 'pplx-embed-v1-0.6b',
						value: 'pplx-embed-v1-0.6b',
						description: '1024 dimensions',
					},
					{
						name: 'pplx-embed-v1-4b',
						value: 'pplx-embed-v1-4b',
						description: '2560 dimensions',
					},
				],
				description:
					'The model which will generate the embeddings. <a href="https://docs.perplexity.ai/docs/embeddings/quickstart">Learn more</a>.',
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
						displayName: 'Batch Size',
						name: 'batchSize',
						default: 512,
						typeOptions: { maxValue: 512 },
						description: 'Maximum number of documents to send in each request',
						type: 'number',
					},
					{
						displayName: 'Strip New Lines',
						name: 'stripNewLines',
						default: true,
						description: 'Whether to strip new lines from the input text',
						type: 'boolean',
					},
					{
						displayName: 'Dimensions',
						name: 'dimensions',
						default: undefined,
						description:
							'The number of dimensions the resulting output embeddings should have. Leave unset to use the model default.',
						type: 'number',
					},
					{
						displayName: 'Timeout',
						name: 'timeout',
						default: -1,
						description:
							'Maximum amount of time a request is allowed to take in seconds. Set to -1 for no timeout.',
						type: 'number',
					},
				],
			},
		],
	};

	async supplyData(this: ISupplyDataFunctions, itemIndex: number): Promise<SupplyData> {
		this.logger.debug('Supply data for Perplexity embeddings');
		const credentials = await this.getCredentials<{ apiKey: string }>('perplexityApi');

		const modelName = this.getNodeParameter('model', itemIndex, 'pplx-embed-v1-0.6b') as string;

		const options = this.getNodeParameter('options', itemIndex, {}) as {
			batchSize?: number;
			stripNewLines?: boolean;
			dimensions?: number;
			timeout?: number;
		};

		if (options.timeout === -1) {
			options.timeout = undefined;
		}

		const configuration: ClientOptions = {
			baseURL: PERPLEXITY_BASE_URL,
			defaultHeaders: { 'X-Source': 'n8n' },
			fetch: aiClientFetch,
			fetchOptions: {
				dispatcher: getProxyAgent(PERPLEXITY_BASE_URL, {}, this.helpers.getSecureEgressFilter()),
			},
		};

		const embeddings = new PerplexityEmbeddings({
			apiKey: credentials.apiKey,
			model: modelName,
			...options,
			configuration,
		});

		return {
			response: logWrapper(embeddings, this),
		};
	}
}
