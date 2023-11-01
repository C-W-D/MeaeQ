import argparse
import sys
import os

this_file_path = sys.path[0]
project_NLP_path = this_file_path
if this_file_path.find('data_generation') != -1 or this_file_path.find('model_steal') != -1:
    project_NLP_path = project_NLP_path + r'/../../'
else:
    project_NLP_path = project_NLP_path + r'/../'
victim_model_version = 'bert_base_uncased' # roberta_base xlnet_base bert_base_uncased
steal_model_version = 'bert_base_uncased' # roberta_base xlnet_base bert_base_uncased


def parse_arguments(parser):
    """
        args
    """
    # parser.add_argument('output_file', type=str, metavar='<output_file>', help='')
    parser.add_argument('--victim_model_version', type=str, default=victim_model_version,
                        help='')
    parser.add_argument('--steal_model_version', type=str, default=steal_model_version,
                        help='')
    parser = bert_config(parser)
    parser = roberta_config(parser)
    parser = xlnet_config(parser)

    parser.add_argument('--bart_large_mnli_path', type=str,
                        default=project_NLP_path + 'pretrained_model/bart_large_mnli'
                        , help='')

    parser.add_argument('--wikitext103_vocab_file', type=str,
                        default=project_NLP_path + './dataset/corpus/wikitext103-vocab.pickle',
                        help='')

    ################################################################################################
    parser.add_argument('--do_data_class_balance', type=bool, default=False, help='')
    parser.add_argument('--weighted_cross_entropy', type=bool, default=False, help='True')
    parser.add_argument('--tokenize_max_length', type=int, default=128, help='')
    if victim_model_version == 'roberta_base':
        parser.add_argument('--visible_device', type=int, default=0, help='CUDA_VISIBLE_DEVICES')
    elif victim_model_version == 'xlnet_base':
        parser.add_argument('--visible_device', type=int, default=1, help='CUDA_VISIBLE_DEVICES')
    else:
        parser.add_argument('--visible_device', type=int, default=0, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num_labels', type=int, default=2, help='class num')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size in dataloader')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate in deep learning')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 regularzation weight')
    parser.add_argument('--optimizer', type=str, default='adam', help='parameter optimizer')
    parser.add_argument('--scheduler', type=str, default='self-definition', help='learning rate update')
    parser.add_argument('--num_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--loss', type=str, default='cross-entropy-loss', help='LOSS')
    # pool and query
    parser.add_argument('--query_num', type=int, default=957, help='')
    parser.add_argument('--seed', type=int, default=56, help='')
    parser.add_argument('--run_seed', type=int, default=31, help='')
    parser.add_argument('--task_name', type=str, default='HATESPEECH',
                        help='SST-2;' +
                             'IMDB;' +
                             'AGNEWS;' +
                             'HATESPEECH;')
    parser.add_argument('--pool_data_source', type=str, default='wiki', help='')
    parser.add_argument('--method', type=str, default='RS',
                        help='RS;' +
                        'DRC;' +
                             'TRF;' +
                             'MeaeQ;' +
                             'AL-RS;' +
                             'AL-US')
    parser.add_argument('--pool_data_type', type=str, default='whole',
                        help='whole;' +
                             'random_subset;' +
                             'reduced_subset_by_prompt;' +
                             'reduced_subset_by_prompt_integrate;')
    parser.add_argument('--initial_sample_method', type=str, default='WIKI',
                        help='random_sentence;' +
                             'data_reduction_kmeans;' +
                             'WIKI;' +
                             'RANDOM;')
    if parser.get_default('pool_data_type') == 'random_subset' or \
            parser.get_default('pool_data_type') == 'whole':
        parser.add_argument('--prompt', type=str, default='None',
                            help='')
    else:
        parser.add_argument('--prompt', type=str, default='This is a hate speech.',
                            help='This is a movie review.' +
                                 'This is a hate speech.')

    if parser.get_default('pool_data_type') == 'reduced_subset_by_prompt' or parser.get_default(
            'pool_data_type') == 'whole' or parser.get_default(
            'pool_data_type') == 'reduced_subset_by_prompt_integrate':
        parser.add_argument('--pool_subsize', type=int, default=-1, help='')
    else:
        parser.add_argument('--pool_subsize', type=int, default=212630, help='')

    # init pool file content
    root = project_NLP_path + './steal/query_and_pool/'
    parser.add_argument('--root', type=str, default=root, help='')

    seed_fn = 'seed=' + str(parser.get_default('seed'))
    task_name_fn = 'task_name=' + parser.get_default('task_name')
    pool_data_source_fn = 'pool_data_source=' + parser.get_default('pool_data_source')
    pool_data_type_fn = 'pool_data_type=' + parser.get_default('pool_data_type')
    prompt_fn = 'prompt=' + parser.get_default('prompt').replace(" ", '-')
    pool_subsize_fn = 'pool_subsize=' + str(parser.get_default('pool_subsize'))

    # build pool file
    root_file = root + seed_fn + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)
    root_file = root_file + task_name_fn + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)
    root_file = root_file + pool_data_source_fn + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)
    root_file = root_file + pool_data_type_fn + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)
    root_file = root_file + prompt_fn + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)
    root_file = root_file + pool_subsize_fn + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)

    parser.add_argument('--pool_data_file', type=str,
                        default=root_file + 'pool_data_sentence.pickle',
                        help='')
    parser.add_argument('--pool_data_pb_file', type=str,
                        default=root_file + 'pool_data_pb.pickle',
                        help='')

    # query
    if parser.get_default('pool_data_type') == 'reduced_subset_by_prompt' or parser.get_default(
            'pool_data_type') == 'reduced_subset_by_prompt_integrate':
        parser.add_argument('--epsilon', type=float, default=0.95, help='')
    else:
        parser.add_argument('--epsilon', type=float, default=-1, help='')


    if parser.get_default('initial_sample_method') == 'random_sentence' \
            or parser.get_default('initial_sample_method') == 'RANDOM' \
            or parser.get_default('initial_sample_method') == 'WIKI':
        parser.add_argument('--initial_drk_model', type=str, default='None',
                            help='sentence-bert;' +
                                 'bart-large-mnli;')
    else:
        parser.add_argument('--initial_drk_model', type=str, default='bart-large-mnli', help='')



    # query file
    epsilone_fn = 'epsilon=' + str(parser.get_default('epsilon'))
    initial_sample_method_fn = 'initial_sample_method=' + parser.get_default('initial_sample_method')
    initial_drk = 'initial_drk_model=' + parser.get_default('initial_drk_model').replace(" ", '-')
    query_num_fn = 'query_num=' + str(parser.get_default('query_num'))
    # build query file
    root_file = root_file + epsilone_fn + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)
    root_file = root_file + initial_sample_method_fn + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)
    root_file = root_file + initial_drk + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)
    root_file = root_file + query_num_fn + '/'
    if not os.path.exists(root_file):
        os.mkdir(root_file)

    parser.add_argument('--query_content_data_file', type=str,
                        default=root_file + 'query_content.pickle',
                        help='')
    parser.add_argument('--query_label_data_file', type=str,
                        default=root_file + 'query_label.pickle',
                        help='')

    ############################################################################################################
    parser.add_argument('--wiki_path', type=str,
                        default=project_NLP_path + './dataset/corpus/wikitext103-sentences.txt',
                        help='')
    parser.add_argument('--imdb_path', type=str,
                        default=project_NLP_path + './dataset/corpus/imdb-sentences.txt',
                        help='')
    parser.add_argument('--sst2_path', type=str,
                        default=project_NLP_path + './dataset/corpus/sst2-sentences.txt',
                        help='')
    parser.add_argument('--iteration_max', type=int, default=-1, help='')
    parser.add_argument('--al_sample_batch_num', type=int, default=20, help='')
    parser.add_argument('--al_sample_method', type=str, default='uncertainty',
                        help='dr-greedy-select-min-max;' +
                             'dr-greedy-select-max-sum;' +
                             'plain-al;' +
                             'topk-diff-probility' +
                             'random;' +
                             'uncertainty')

    parser.add_argument('--topk', type=float, default=1.0, help='top k rate')
    parser.add_argument('--get_candidate_set', type=bool, default=False, help='')
    parser.add_argument('--candidate_set_topk', type=float, default=0.01, help='top k rate')

    return parser


def bert_config(parser):
    parser.add_argument('--project_NLP_path', type=str, default=project_NLP_path,
                        help='')
    parser.add_argument('--bert_hidden_size', type=int, default=768, help='')
    parser.add_argument('--bert_hidden_dropout_prob', type=float, default=0.2, help='')
    parser.add_argument('--bert_output_hidden_states', type=bool, default=True, help='')
    parser.add_argument('--steal_bert_path', type=str, default=project_NLP_path + 'pretrained_model/' + "bert_base_uncased",
                        help='')
    parser.add_argument('--steal_bert_vocab_path', type=str,
                        default=project_NLP_path + 'pretrained_model/' + "bert_base_uncased" + '/'
                        , help='')
    parser.add_argument('--bpc_dropout_prob', type=float, default=0.2, help='')
    parser.add_argument('--victim_bert_path', type=str,
                        default=project_NLP_path + 'pretrained_model/' + "bert_base_uncased"
                        , help='')
    #     print(project_NLP_path + 'pretrained_model/' + "bert_base_uncased" + '/')
    parser.add_argument('--victim_bert_vocab_path', type=str,
                        default=project_NLP_path + 'pretrained_model/' + "bert_base_uncased" + '/'
                        , help='')
    parser.add_argument('--do_lower_case', type=bool, default=True, help='')
    parser.add_argument('--saved_model_path', type=str, default=project_NLP_path + './saved_model/',
                        help='')
    parser.add_argument('--victim_model_checkpoint', type=str, default='-' + victim_model_version + '-victim-model.pkl',
                        help='')
    parser.add_argument('--steal_model_checkpoint', type=str, default='-' + steal_model_version + '-steal-model.pkl',
                        help='')

    return parser

def roberta_config(parser):
    parser.add_argument('--roberta_hidden_size', type=int, default=768, help='')
    parser.add_argument('--roberta_hidden_dropout_prob', type=float, default=0.1, help='')
    parser.add_argument('--roberta_output_hidden_states', type=bool, default=True, help='')
    parser.add_argument('--steal_roberta_path', type=str, default=project_NLP_path + 'pretrained_model/' + "roberta_base",
                        help='')
    parser.add_argument('--steal_roberta_vocab_path', type=str,
                        default=project_NLP_path + 'pretrained_model/' + "roberta_base" + '/'
                        , help='')
    parser.add_argument('--rpc_dropout_prob', type=float, default=0.1, help='')
    parser.add_argument('--victim_roberta_path', type=str,
                        default=project_NLP_path + 'pretrained_model/' + "roberta_base"
                        , help='')
    #     print(project_NLP_path + 'pretrained_model/' + "roberta_base" + '/')
    parser.add_argument('--victim_roberta_vocab_path', type=str,
                        default=project_NLP_path + 'pretrained_model/' + "roberta_base" + '/'
                        , help='')
    return parser

def xlnet_config(parser):
    parser.add_argument('--xlnet_hidden_size', type=int, default=768, help='')
    parser.add_argument('--xlnet_hidden_dropout_prob', type=float, default=0.1, help='')
    parser.add_argument('--xlnet_output_hidden_states', type=bool, default=True, help='')
    parser.add_argument('--steal_xlnet_path', type=str, default=project_NLP_path + 'pretrained_model/' + "xlnet_base",
                        help='')
    parser.add_argument('--steal_xlnet_vocab_path', type=str,
                        default=project_NLP_path + 'pretrained_model/' + "xlnet_base" + '/'
                        , help='')
    parser.add_argument('--xpc_dropout_prob', type=float, default=0.2, help='')
    parser.add_argument('--victim_xlnet_path', type=str,
                        default=project_NLP_path + 'pretrained_model/' + "xlnet_base"
                        , help='')
    parser.add_argument('--victim_xlnet_vocab_path', type=str,
                        default=project_NLP_path + 'pretrained_model/' + "xlnet_base" + '/'
                        , help='')
    return parser

class ArgParser:
    def __init__(self):
        parse = argparse.ArgumentParser(description='')
        self.parser = parse_arguments(parse)

    def get_parser(self):
        return self.parser.parse_args()
#         return self.parser.parse_args(args=[])
    def get_parser_jupyter(self):
        return self.parser.parse_args(args=[])

