retrieval_accuracy = {
    "BM25": {
        'NQ': {1: 0.2385, 3: 0.3935, 5: 0.47, 10: 0.5735, 20: 0.6605, 50: 0.7555, 100: 0.801},
        'TriviaQA': {1: 0.554, 3: 0.728, 5: 0.7865, 10: 0.865, 20: 0.914, 50: 0.9565, 100: 0.975},
        'hotpotQA': {1: 0.277, 3: 0.4225, 5: 0.47, 10: 0.534, 20: 0.5915, 50: 0.6705, 100: 0.721},
        'conala': {1: 0.0398, 3: 0.0625, 5: 0.0909, 10: 0.1364, 20: 0.1364, 50: 0.2273, 100: 0.2973},
        'pandas_numpy_eval': {1: 0.0357, 3: 0.0883, 5: 0.13, 10: 0.1538, 20: 0.2143, 50: 0.2912, 100: 0.3368},
        'DS1000': {1: 0.0372, 3: 0.0588, 5: 0.0704, 10: 0.0891, 20: 0.1283, 50: 0.1776, 100: 0.2049},
    },
    "miniLM": {
        'NQ': {1: 0.3935, 3: 0.5865, 5: 0.674, 10: 0.7665, 20: 0.8185, 50: 0.882, 100: 0.9185},
        'TriviaQA': {1: 0.503, 3: 0.707, 5: 0.7755, 10: 0.849, 20: 0.9015, 50: 0.942, 100: 0.9685},
        'hotpotQA': {1: 0.248, 3: 0.36, 5: 0.402, 10: 0.453, 20: 0.503, 50: 0.567, 100: 0.61},
        'conala': {1: 0.0341, 3: 0.0777, 5: 0.1061, 10: 0.1913, 20: 0.2775, 50: 0.4129, 100: 0.4678},
        'pandas_numpy_eval': {1: 0.0565, 3: 0.1389, 5: 0.1788, 10: 0.2527, 20: 0.3142, 50: 0.4407, 100: 0.5141},
        'DS1000': {1: 0.0499, 3: 0.0807, 5: 0.1094, 10: 0.1611, 20: 0.1993, 50: 0.2908, 100: 0.3431},
    },
    "openai-embedding": {
        'NQ': {1: 0.472, 3: 0.6635, 5: 0.7345, 10: 0.8155, 20: 0.8705, 50: 0.9225, 100: 0.9485},
        'TriviaQA': {1: 0.6275, 3: 0.8175, 5: 0.871, 10: 0.9175, 20: 0.9515, 50: 0.9785, 100: 0.989},
        'hotpotQA': {1: 0.351, 3: 0.532, 5: 0.582, 10: 0.639, 20: 0.688, 50: 0.742, 100: 0.78},
        'conala': {1: 0.004, 3: 0.0873, 5: 0.0992, 10: 0.1508, 20: 0.2113, 50: 0.3075, 100: 0.3661},
        'pandas_numpy_eval': {1: 0.0987, 3: 0.1761, 5: 0.2207, 10: 0.2924, 20: 0.3926, 50: 0.4729, 100: 0.5572},
        'DS1000': {1: 0.0434, 3: 0.0878, 5: 0.1169, 10: 0.1809, 20: 0.2275, 50: 0.3031, 100: 0.3605},
    },
    "openai-embedding_rerank_cohere": {
        'NQ': None,
        'TriviaQA': None,
        'hotpotQA': {1: 0.419, 3: 0.646, 5: 0.693, 10: 0.732, 20: 0.758, 50: 0.776, 100: 0.78},
        'conala': {1: 0.0278, 3: 0.0615, 5: 0.1091, 10: 0.1855, 20: 0.2331, 50: 0.2708, 100: 0.3631},
        'pandas_numpy_eval': {1: 0.1407, 3: 0.266, 5: 0.3179, 10: 0.3827, 20: 0.4401, 50: 0.519, 100: 0.5606},
        'DS1000': {1: 0.0522, 3: 0.0833, 5: 0.1085, 10: 0.1561, 20: 0.2127, 50: 0.2848, 100: 0.3631},
    },
    "contriever": {
        'NQ': {1: 0.189, 3: 0.3695, 5: 0.4705, 10: 0.586, 20: 0.6875, 50: 0.792, 100: 0.8495},
        'TriviaQA': {1: 0.3975, 3: 0.6155, 5: 0.714, 10: 0.799, 20: 0.881, 50: 0.94, 100: 0.969},
        'hotpotQA': {1: 0.237, 3: 0.348, 5: 0.403, 10: 0.472, 20: 0.534, 50: 0.609, 100: 0.663},
        'conala': {1: 0.017, 3: 0.0549, 5: 0.0663, 10: 0.0928, 20: 0.0928, 50: 0.161, 100: 0.215},
        'pandas_numpy_eval': {1: 0.0119, 3: 0.0268, 5: 0.0298, 10: 0.0595, 20: 0.0923, 50: 0.1949, 100: 0.255},
        'DS1000': {1: 0.0133, 3: 0.0334, 5: 0.0494, 10: 0.0545, 20: 0.0656, 50: 0.122, 100: 0.1668},
    },
    "codeT5": {
        'NQ': None,
        'TriviaQA': None,
        'hotpotQA': None,
        'conala': {1: 0.0739, 3: 0.1136, 5: 0.1402, 10: 0.1705, 20: 0.2973, 50: 0.447, 100: 0.5767},
        'pandas_numpy_eval': {1: 0.006, 3: 0.0179, 5: 0.0476, 10: 0.0709, 20: 0.1334, 50: 0.2021, 100: 0.3079},
        'DS1000': {1: 0.0064, 3: 0.0223, 5: 0.0244, 10: 0.0295, 20: 0.0433, 50: 0.0918, 100: 0.1422},
    }
}


# 'BM25': {'em': 0.6315, 'f1': 0.7166204887218547, 'prec': 0.6926273072795384, 'recall': 0.8031583333333334},
#         'miniLM': {'em': 0.0525, 'f1': 0.1984660821533895, 'prec': 0.14775189912621844, 'recall': 0.6317416666666666},
#         'openai-embedding': {'em': 0.0635, 'f1': 0.21117684822958296, 'prec': 0.16022261271110894, 'recall': 0.6570000000000006},
#         'contriever': {'em': 0.044, 'f1': 0.17035605564717382, 'prec': 0.1270591187148195, 'recall': 0.5581333333333336}

retriever_perf_llama = {
    'NQ': {
        # 'BM25': {'em': 0.05, 'f1': 0.179, 'prec': 0.135, 'recall': 0.546},
        'BM25-5': {'em': 0.031, 'f1': 0.154, 'prec': 0.109, 'recall': 0.516, 'has_answer': 0.423, 'prompt_length': 929.465, 'perplexity': 1.071},
        # 'miniLM': {'em': 0.646, 'f1': 0.724, 'prec': 0.703, 'recall': 0.808},
        'openai-embedding': {'em': 0.064, 'f1': 0.211, 'prec': 0.16, 'recall': 0.657, 'has_answer': 0.559, 'prompt_length': 1720.924, 'perplexity': 1.076},
        # 'contriever': {'em': 0.633, 'f1': 0.7104561200428963, 'prec': 0.6893711013373296, 'recall': 0.7928833333333333}
    },
    'TriviaQA': {
        # 'BM25': {'em': 0.1915, 'f1': 0.366, 'prec': 0.297, 'recall': 0.842},
        'BM25-5': {'em': 0.135, 'f1': 0.317, 'prec': 0.245, 'recall': 0.814, 'has_answer': 0.766, 'ret_recall': 0.786, 'oracle_percent': 0.435, 'oracle_rank': 1.556, 'prompt_length': 955.889, 'perplexity': 1.062},
        # 'miniLM': {'em': 0.1745, 'f1': 0.3471259913860253, 'prec': 0.27950000384825197, 'recall': 0.8294466450216446},
        'openai-embedding': {'em': 0.202, 'f1': 0.378, 'prec': 0.309, 'recall': 0.861, 'has_answer': 0.825, 'prompt_length': 1759.983, 'perplexity': 1.068},
        # 'contriever': {'em': 0.141, 'f1': 0.32077085408830275, 'prec': 0.2508410955069186, 'recall': 0.8228712301587301}
    },
    'hotpotQA': {
        # 'BM25': {'em': 0.0505, 'f1': 0.174, 'prec': 0.131, 'recall': 0.533},
        'BM25-5': {'em': 0.087, 'f1': 0.208, 'prec': 0.17, 'recall': 0.489, 'has_answer': 0.394, 'ret_recall': 0.47, 'oracle_percent': 0.188, 'oracle_rank': 1.761, 'prompt_length': 557.407, 'perplexity': 1.062},
        # 'miniLM': {'em': 0.0505, 'f1': 0.15961374764570338, 'prec': 0.12033470411281898, 'recall': 0.4730590326340328},
        'openai-embedding': {'em': 0.046, 'f1': 0.169, 'prec': 0.125, 'recall': 0.536, 'has_answer': 0.443, 'prompt_length': 1150.41, 'perplexity': 1.06},
        # 'contriever': {'em': 0.034, 'f1': 0.1464155973383291, 'prec': 0.10548973484569028, 'recall': 0.4808009157509159}
    },
    'conala': {
        'BM25': {'pass@1': 0.19, 'prompt_length': 2999.786, 'perplexity': 1.132, 'retrieval_consistency': 0.548, 'syntax_error_percent': 0.226, 'semantic_error_percent': 0.5},
        # 'miniLM': {'pass@1': 0.214, 'ret_recall': 0.111, 'oracle_percent': 0.029, 'oracle_rank': 2.417, 'prompt_length': 732.595, 'perplexity': 1.13, 'retrieval_consistency': 1.25, 'syntax_error_percent': 0.179, 'semantic_error_percent': 0.488},
        'openai-embedding': {'pass@1': 0.226, 'prompt_length': 542.631, 'perplexity': 1.129, 'retrieval_consistency': 0.738, 'syntax_error_percent': 0.226, 'semantic_error_percent': 0.44},
        # 'codeT5': {'pass@1': 0.238, 'ret_recall': 0.147, 'oracle_percent': 0.038, 'oracle_rank': 2.125, 'prompt_length': 1232.524, 'perplexity': 1.121, 'retrieval_consistency': 1.083, 'syntax_error_percent': 0.179, 'semantic_error_percent': 0.429},
    },
    'DS1000': {
        'BM25': {'pass@1': 0.084, 'prompt_length': 4851.369, 'perplexity': 1.136, 'retrieval_consistency': 0.503, 'syntax_error_percent': 0.121, 'semantic_error_percent': 0.439},
        # 'miniLM': {'pass@1': 0.06, 'ret_recall': 0.109, 'oracle_percent': 0.042, 'oracle_rank': 2.121, 'prompt_length': 2582.197, 'perplexity': 1.143, 'retrieval_consistency': 1.541, 'syntax_error_percent': 0.14, 'semantic_error_percent': 0.427},
        'openai-embedding': {'pass@1': 0.095, 'prompt_length': 3275.248, 'perplexity': 1.138, 'retrieval_consistency': 1.268, 'syntax_error_percent': 0.134, 'semantic_error_percent': 0.401},
        # 'codeT5': {'pass@1': 0.102, 'ret_recall': 0.024, 'oracle_percent': 0.006, 'oracle_rank': 2.6, 'prompt_length': 4172.554, 'perplexity': 1.137, 'retrieval_consistency': 0.459, 'syntax_error_percent': 0.127, 'semantic_error_percent': 0.414},
    },
    'pandas_numpy_eval': {
        'BM25': {'pass@1': 0.629, 'prompt_length': 3984.994, 'perplexity': 1.136, 'retrieval_consistency': 0.234, 'syntax_error_percent': 0.06, 'semantic_error_percent': 0.305},
        # 'miniLM': {'pass@1': 0.521, 'ret_recall': 0.18, 'oracle_percent': 0.046, 'oracle_rank': 2.684, 'prompt_length': 1640.132, 'perplexity': 1.136, 'retrieval_consistency': 1.192, 'syntax_error_percent': 0.054, 'semantic_error_percent': 0.317},
        'openai-embedding': {'pass@1': 0.551, 'prompt_length': 1935.874, 'perplexity': 1.135, 'retrieval_consistency': 0.964, 'syntax_error_percent': 0.042, 'semantic_error_percent': 0.323}
        # 'codeT5': {'pass@1': 0.515, 'ret_recall': 0.048, 'oracle_percent': 0.012, 'oracle_rank': 3.5, 'prompt_length': 2107.778, 'perplexity': 1.139, 'retrieval_consistency': 0.719, 'syntax_error_percent': 0.09, 'semantic_error_percent': 0.341},
    }
}


retriever_perf_gpt = {
    'NQ': {
        # 'BM25': {'em': 0.254, 'f1': 0.38, 'prec': 0.372, 'recall': 0.479},
        'BM25-5': {'em': 0.227, 'f1': 0.34, 'prec': 0.334, 'recall': 0.434, 'has_answer': 0.359, 'ret_recall': 0.47, 'oracle_percent': 0.176, 'oracle_rank': 2.017, 'prompt_length': 782.45, 'perplexity': 1.045},
        # 'miniLM': {'em': 0.319, 'f1': 0.45539919755019664, 'prec': 0.43953505825638856, 'recall': 0.5902750000000003},
        'openai-embedding': {'em': 0.344, 'f1': 0.489, 'prec': 0.475, 'recall': 0.623, 'has_answer': 0.543, 'ret_recall': 0.816, 'oracle_percent': 0.242, 'oracle_rank': 2.277, 'prompt_length': 1460.296, 'perplexity': 1.039},
        # 'contriever': {'em': 0.267, 'f1': 0.39120396494624526, 'prec': 0.38556238871240267, 'recall': 0.4977666666666666}
    },
    'TriviaQA': {
        # 'BM25': {'em': 0.6305, 'f1': 0.717, 'prec': 0.693, 'recall': 0.803},
        'BM25-5': {'em': 0.608, 'f1': 0.691, 'prec': 0.668, 'recall': 0.777, 'has_answer': 0.752, 'ret_recall': 0.786, 'oracle_percent': 0.435, 'oracle_rank': 1.556, 'prompt_length': 805.601, 'perplexity': 1.025},
        # 'miniLM': {'em': 0.646, 'f1': 0.7243253494273003, 'prec': 0.7028446331778762, 'recall': 0.8075787878787877},
        'openai-embedding': {'em': 0.67, 'f1': 0.752, 'prec': 0.729, 'recall': 0.837, 'has_answer': 0.818, 'prompt_length': 1500.321, 'perplexity': 1.023},
        # 'contriever': {'em': 0.633, 'f1': 0.7104561200428963, 'prec': 0.6893711013373296, 'recall': 0.7928833333333333}
    },
    'hotpotQA': {
        # 'BM25': {'em': 0.2905, 'f1': 0.414, 'prec': 0.419, 'recall': 0.464},
        'BM25-5': {'em': 0.274, 'f1': 0.393, 'prec': 0.399, 'recall': 0.441, 'has_answer': 0.384, 'ret_recall': 0.47, 'oracle_percent': 0.188, 'oracle_rank': 1.761, 'prompt_length': 458.979, 'perplexity': 1.036},
        # 'miniLM': {'em': 0.241, 'f1': 0.3584158256474385, 'prec': 0.36329800027556275, 'recall': 0.4102590298590299},
        'openai-embedding': {'em': 0.295, 'f1': 0.421, 'prec': 0.426, 'recall': 0.475, 'has_answer': 0.423, 'prompt_length': 963.867, 'perplexity': 1.039},
        # 'contriever': {'em': 0.263, 'f1': 0.380517603212683, 'prec': 0.3865593114832381, 'recall': 0.43119434731934764}
    },
    'conala': {
        'BM25': {'pass@1': 0.357, 'prompt_length': 2958.607, 'perplexity': 1.044, 'retrieval_consistency': 0.667, 'syntax_error_percent': 0.143, 'semantic_error_percent': 0.381},
        # 'miniLM': {'pass@1': 0.298, 'ret_recall': 0.111, 'oracle_percent': 0.029, 'oracle_rank': 2.417, 'prompt_length': 663.631},
        'openai-embedding': {'pass@1': 0.286, 'prompt_length': 484.774, 'perplexity': 1.046, 'retrieval_consistency': 0.81, 'syntax_error_percent': 0.202, 'semantic_error_percent': 0.405},
        # 'codeT5': {'pass@1': 0.345, 'ret_recall': 0.147, 'oracle_percent': 0.038, 'oracle_rank': 2.125, 'prompt_length': 1151.321}
    },
    'DS1000': {
        'BM25': {'pass@1': 0.353, 'prompt_length': 4806.331, 'perplexity': 1.041, 'retrieval_consistency': 0.312, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.363},
        # 'miniLM': {'pass@1': 0.276, 'ret_recall': 0.109, 'oracle_percent': 0.042, 'oracle_rank': 2.121, 'prompt_length': 2360.783},
        'openai-embedding': {'pass@1': 0.339, 'prompt_length': 3139.516, 'perplexity': 1.039, 'retrieval_consistency': 0.911, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.369},
        # 'codeT5': {'pass@1': 0.306, 'ret_recall': 0.024, 'oracle_percent': 0.006, 'oracle_rank': 2.6, 'prompt_length': 3971.35}
    },
    'pandas_numpy_eval': {
        'BM25': {'pass@1': 0.778, 'prompt_length': 3831.647, 'perplexity': 1.038, 'retrieval_consistency': 0.246, 'syntax_error_percent': 0.012, 'semantic_error_percent': 0.263},
        # 'miniLM': {'pass@1': 0.725, 'ret_recall': 0.18, 'oracle_percent': 0.046, 'oracle_rank': 2.684, 'prompt_length': 1478.94},
        'openai-embedding': {'pass@1': 0.719, 'prompt_length': 1805.76, 'perplexity': 1.037, 'retrieval_consistency': 0.928, 'syntax_error_percent': 0.042, 'semantic_error_percent': 0.287},
        # 'codeT5': {'pass@1': 0.772, 'ret_recall': 0.048, 'oracle_percent': 0.012, 'oracle_rank': 3.5, 'prompt_length': 1962.892}
    }
}


code_ret_recall_llama_n_10 = {
    "conala": {
        1: {'pass@1': 0.2880952380952381, 'pass@3': 0.4255952380952382, 'pass@5': 0.4743008314436886, 'pass@10': 0.5238095238095238},
        0.8: {'pass@1': 0.2428571428571428, 'pass@3': 0.3691468253968254, 'pass@5': 0.4118008314436887, 'pass@10': 0.4523809523809524},
        0.6: {'pass@1': 0.21666666666666662, 'pass@3': 0.3387896825396826, 'pass@5': 0.39828987150415734, 'pass@10': 0.47619047619047616},
        0.4: {'pass@1': 0.1845238095238095, 'pass@3': 0.2881944444444445, 'pass@5': 0.3360733182161755, 'pass@10': 0.38095238095238093},
        0.2: {'pass@1': 0.15238095238095237, 'pass@3': 0.2441468253968254, 'pass@5': 0.2829743008314437, 'pass@10': 0.32142857142857145},
        0: {'pass@1': 0.14047619047619048, 'pass@3': 0.2147817460317461, 'pass@5': 0.2543461829176116, 'pass@10': 0.2976190476190476},
        'none': {'pass@1': 0.183, 'pass@3': 0.276, 'pass@5': 0.32, 'pass@10': 0.393, 'prompt_length': 92.714, 'perplexity': 1.121}
    },
    "DS1000": {
        1: {'pass@1': 0.15615917581020558, 'pass@3': 0.27813482400724965, 'pass@5': 0.34162171345619097, 'pass@10': 0.41703583093743274},
        0.8: {'pass@1': 0.1379595728451564, 'pass@3': 0.2584909045421059, 'pass@5': 0.33676267848178604, 'pass@10': 0.4604490602202273},
        0.6: {'pass@1': 0.12353796279997653, 'pass@3': 0.21912760696799594, 'pass@5': 0.2720179575347386, 'pass@10': 0.35867218212755975},
        0.4: {'pass@1': 0.14027885348823563, 'pass@3': 0.245518357976882, 'pass@5': 0.3070477875445457, 'pass@10': 0.4004366406540319},
        0.2: {'pass@1': 0.11634947876938724, 'pass@3': 0.2059071254408783, 'pass@5': 0.25494190581433146, 'pass@10': 0.3343694380879736},
        0: {'pass@1': 0.08520829666138591, 'pass@3': 0.1856552136100191, 'pass@5': 0.24744155345318583, 'pass@10': 0.343789727943046},
        'none': {'pass@1': 0.153, 'pass@3': 0.268, 'pass@5': 0.327, 'pass@10': 0.409, 'prompt_length': 514.739, 'perplexity': 1.121}
    },
    "pandas_numpy_eval": {
        1: {'pass@1': 0.5347305389221557, 'pass@3': 0.6707085828343313, 'pass@5': 0.7136679022906568, 'pass@10': 0.7544910179640718},
        0.8: {'pass@1': 0.5203592814371256, 'pass@3': 0.6723053892215566, 'pass@5': 0.7289468681684251, 'pass@10': 0.7844311377245509},
        0.6: {'pass@1': 0.47724550898203605, 'pass@3': 0.6163672654690615, 'pass@5': 0.672036878623705, 'pass@10': 0.7305389221556886},
        0.4: {'pass@1': 0.4574850299401197, 'pass@3': 0.6074351297405187, 'pass@5': 0.6651696606786425, 'pass@10': 0.7245508982035929},
        0.2: {'pass@1': 0.42814371257485045, 'pass@3': 0.5779441117764469, 'pass@5': 0.6364176409086588, 'pass@10': 0.7005988023952096},
        0: {'pass@1': 0.39341317365269457, 'pass@3': 0.5392215568862274, 'pass@5': 0.5979232012166144, 'pass@10': 0.6586826347305389},
        'none': {'pass@1': 0.55, 'pass@3': 0.681, 'pass@5': 0.725, 'pass@10': 0.772, 'prompt_length': 187.036, 'perplexity': 1.077}
    }
}


code_ret_recall_llama_n_1 = {
    "conala": {
        1: {'pass@1': 0.298, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.589, 'prompt_length': 853.226, 'perplexity': 1.119, 'retrieval_consistency': 1.0, 'syntax_error_percent': 0.143, 'semantic_error_percent': 0.19},
        0.8: {'pass@1': 0.274, 'ret_recall': 0.794, 'oracle_percent': 0.794, 'oracle_rank': 1.602, 'prompt_length': 729.262, 'perplexity': 1.113, 'retrieval_consistency': 0.905, 'syntax_error_percent': 0.143, 'semantic_error_percent': 0.226},
        0.6: {'pass@1': 0.238, 'ret_recall': 0.599, 'oracle_percent': 0.599, 'oracle_rank': 1.586, 'prompt_length': 601.821, 'perplexity': 1.118, 'retrieval_consistency': 0.726, 'syntax_error_percent': 0.19, 'semantic_error_percent': 0.274},
        0.4: {'pass@1': 0.19, 'ret_recall': 0.389, 'oracle_percent': 0.389, 'oracle_rank': 1.614, 'prompt_length': 509.131, 'perplexity': 1.124, 'retrieval_consistency': 0.607, 'syntax_error_percent': 0.19, 'semantic_error_percent': 0.345},
        0.2: {'pass@1': 0.167, 'ret_recall': 0.198, 'oracle_percent': 0.198, 'oracle_rank': 1.633, 'prompt_length': 377.0, 'perplexity': 1.13, 'retrieval_consistency': 0.536, 'syntax_error_percent': 0.19, 'semantic_error_percent': 0.405},
        0: {'pass@1': 0.167, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 217.095, 'perplexity': 1.131, 'retrieval_consistency': 0.44, 'syntax_error_percent': 0.202, 'semantic_error_percent': 0.476}
    },
    "DS1000": {
        1: {'pass@1': 0.199, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 2.159, 'prompt_length': 2270.185, 'perplexity': 1.143, 'retrieval_consistency': 1.21, 'syntax_error_percent': 0.102, 'semantic_error_percent': 0.191},
        0.8: {'pass@1': 0.175, 'ret_recall': 0.799, 'oracle_percent': 0.799, 'oracle_rank': 2.178, 'prompt_length': 2215.07, 'perplexity': 1.145, 'retrieval_consistency': 1.191, 'syntax_error_percent': 0.076, 'semantic_error_percent': 0.229},
        0.6: {'pass@1': 0.179, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 2.139, 'prompt_length': 2123.834, 'perplexity': 1.151, 'retrieval_consistency': 1.172, 'syntax_error_percent': 0.07, 'semantic_error_percent': 0.28},
        0.4: {'pass@1': 0.159, 'ret_recall': 0.399, 'oracle_percent': 0.399, 'oracle_rank': 2.103, 'prompt_length': 2084.478, 'perplexity': 1.154, 'retrieval_consistency': 1.032, 'syntax_error_percent': 0.096, 'semantic_error_percent': 0.312},
        0.2: {'pass@1': 0.152, 'ret_recall': 0.194, 'oracle_percent': 0.194, 'oracle_rank': 2.134, 'prompt_length': 2020.943, 'perplexity': 1.154, 'retrieval_consistency': 0.904, 'syntax_error_percent': 0.127, 'semantic_error_percent': 0.325},
        0: {'pass@1': 0.124, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 2024.369, 'perplexity': 1.155, 'retrieval_consistency': 0.783, 'syntax_error_percent': 0.102, 'semantic_error_percent': 0.433}
    },
    "pandas_numpy_eval": {
        1: {'pass@1': 0.599, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.427, 'prompt_length': 1223.281, 'perplexity': 1.124, 'retrieval_consistency': 0.964, 'syntax_error_percent': 0.054, 'semantic_error_percent': 0.198},
        0.8: {'pass@1': 0.563, 'ret_recall': 0.795, 'oracle_percent': 0.795, 'oracle_rank': 1.42, 'prompt_length': 1096.006, 'perplexity': 1.125, 'retrieval_consistency': 0.796, 'syntax_error_percent': 0.054, 'semantic_error_percent': 0.222},
        0.6: {'pass@1': 0.557, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 1.348, 'prompt_length': 996.479, 'perplexity': 1.128, 'retrieval_consistency': 0.677, 'syntax_error_percent': 0.066, 'semantic_error_percent': 0.257},
        0.4: {'pass@1': 0.545, 'ret_recall': 0.395, 'oracle_percent': 0.395, 'oracle_rank': 1.326, 'prompt_length': 900.713, 'perplexity': 1.129, 'retrieval_consistency': 0.545, 'syntax_error_percent': 0.084, 'semantic_error_percent': 0.305},
        0.2: {'pass@1': 0.521, 'ret_recall': 0.199, 'oracle_percent': 0.199, 'oracle_rank': 1.2, 'prompt_length': 779.413, 'perplexity': 1.131, 'retrieval_consistency': 0.491, 'syntax_error_percent': 0.078, 'semantic_error_percent': 0.329},
        0: {'pass@1': 0.491, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 691.868, 'perplexity': 1.13, 'retrieval_consistency': 0.407, 'syntax_error_percent': 0.084, 'semantic_error_percent': 0.359}
    }
}


code_ret_recall_gpt_n_1 = {
    "conala": {
        1: {'pass@1': 0.381, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.589, 'prompt_length': 820.869, 'perplexity': 1.035, 'retrieval_consistency': 0.94, 'syntax_error_percent': 0.143, 'semantic_error_percent': 0.262},
        0.8: {'pass@1': 0.333, 'ret_recall': 0.794, 'oracle_percent': 0.794, 'oracle_rank': 1.602, 'prompt_length': 691.393, 'perplexity': 1.039, 'retrieval_consistency': 0.833, 'syntax_error_percent': 0.179, 'semantic_error_percent': 0.31},
        0.6: {'pass@1': 0.321, 'ret_recall': 0.599, 'oracle_percent': 0.599, 'oracle_rank': 1.586, 'prompt_length': 562.964, 'perplexity': 1.04, 'retrieval_consistency': 0.714, 'syntax_error_percent': 0.202, 'semantic_error_percent': 0.333},
        0.4: {'pass@1': 0.286, 'ret_recall': 0.389, 'oracle_percent': 0.389, 'oracle_rank': 1.614, 'prompt_length': 471.881, 'perplexity': 1.042, 'retrieval_consistency': 0.583, 'syntax_error_percent': 0.238, 'semantic_error_percent': 0.357},
        0.2: {'pass@1': 0.238, 'ret_recall': 0.198, 'oracle_percent': 0.198, 'oracle_rank': 1.633, 'prompt_length': 339.857, 'perplexity': 1.047, 'retrieval_consistency': 0.548, 'syntax_error_percent': 0.286, 'semantic_error_percent': 0.429},
        0: {'pass@1': 0.214, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 178.619, 'perplexity': 1.046, 'retrieval_consistency': 0.417, 'syntax_error_percent': 0.298, 'semantic_error_percent': 0.44}
    },
    'DS1000': {
        1: {'pass@1': 0.37, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 2.159, 'prompt_length': 2172.236, 'perplexity': 1.034, 'retrieval_consistency': 1.268, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.197},
        0.8: {'pass@1': 0.369, 'ret_recall': 0.799, 'oracle_percent': 0.799, 'oracle_rank': 2.178, 'prompt_length': 2111.662, 'perplexity': 1.039, 'retrieval_consistency': 1.146, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.255},
        0.6: {'pass@1': 0.342, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 2.139, 'prompt_length': 2009.433, 'perplexity': 1.037, 'retrieval_consistency': 1.032, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.293},
        # 0.4: {'pass@1': 0.366, 'ret_recall': 0.399, 'oracle_percent': 0.399, 'oracle_rank': 2.103, 'prompt_length': 1969.465, 'perplexity': 1.04, 'retrieval_consistency': 0.783, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.35},
        0.4: {'pass@1': 0.366, 'ret_recall': 0.399, 'oracle_percent': 0.399, 'oracle_rank': 2.103, 'prompt_length': 1969.465, 'perplexity': 1.04, 'retrieval_consistency': 0.783, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.35},
        0.2: {'pass@1': 0.298, 'ret_recall': 0.194, 'oracle_percent': 0.194, 'oracle_rank': 2.134, 'prompt_length': 1900.93, 'perplexity': 1.04, 'retrieval_consistency': 0.637, 'syntax_error_percent': 0.013, 'semantic_error_percent': 0.376},
        0: {'pass@1': 0.29, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 1898.79, 'perplexity': 1.039, 'retrieval_consistency': 0.561, 'syntax_error_percent': 0.019, 'semantic_error_percent': 0.389}
    },
    'pandas_numpy_eval': {
        1: {'pass@1': 0.778, 'prompt_length': 1164.731, 'perplexity': 1.027, 'retrieval_consistency': 1.108, 'syntax_error_percent': 0.036, 'semantic_error_percent': 0.126},
        0.8: {'pass@1': 0.766, 'prompt_length': 1032.581, 'perplexity': 1.028, 'retrieval_consistency': 0.904, 'syntax_error_percent': 0.036, 'semantic_error_percent': 0.174},
        0.6: {'pass@1': 0.754, 'prompt_length': 932.545, 'perplexity': 1.028, 'retrieval_consistency': 0.743, 'syntax_error_percent': 0.012, 'semantic_error_percent': 0.228},
        0.4: {'pass@1': 0.749, 'prompt_length': 837.796, 'perplexity': 1.033, 'retrieval_consistency': 0.599, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.275},
        0.2: {'pass@1': 0.695, 'prompt_length': 711.078, 'perplexity': 1.035, 'retrieval_consistency': 0.473, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.305},
        0: {'pass@1': 0.671, 'prompt_length': 619.617, 'perplexity': 1.036, 'retrieval_consistency': 0.371, 'syntax_error_percent': 0.012, 'semantic_error_percent': 0.353}
    }
}

code_ret_doc_type_llama_n_1 = {
    "conala": {
        "oracle": {'pass@1': 0.298, 'prompt_length': 853.226, 'perplexity': 1.119, 'retrieval_consistency': 1.0, 'syntax_error_percent': 0.143, 'semantic_error_percent': 0.19},
        # "retrieved": {'pass@1': 0.16666666666666666},
        "distracting": {'pass@1': 0.167, 'prompt_length': 217.095, 'perplexity': 1.131, 'retrieval_consistency': 0.429, 'syntax_error_percent': 0.202, 'semantic_error_percent': 0.464},
        "random": {'pass@1': 0.238, 'prompt_length': 449.75, 'perplexity': 1.131, 'retrieval_consistency': 0.024, 'syntax_error_percent': 0.226, 'semantic_error_percent': 0.429},
        "irrelevant_dummy": {'pass@1': 0.25, 'prompt_length': 912.119, 'perplexity': 1.11, 'syntax_error_percent': 0.238, 'semantic_error_percent': 0.369},
        "irrelevant_diff": {'pass@1': 0.19, 'prompt_length': 912.119, 'perplexity': 1.123, 'syntax_error_percent': 0.286, 'semantic_error_percent': 0.417},
        "none": {'pass@1': 0.226, 'prompt_length': 92.714, 'perplexity': 1.137, 'syntax_error_percent': 0.298, 'semantic_error_percent': 0.417},
    },
    "DS1000": {
        "oracle": {'pass@1': 0.199, 'prompt_length': 2270.185, 'perplexity': 1.143, 'retrieval_consistency': 1.204, 'syntax_error_percent': 0.102, 'semantic_error_percent': 0.191},
        # "retrieved": {'pass@1': 0.08371569950517317},
        "distracting": {'pass@1': 0.123, 'prompt_length': 2024.369, 'perplexity': 1.153, 'retrieval_consistency': 0.764, 'syntax_error_percent': 0.115, 'semantic_error_percent': 0.414},
        "random": {'pass@1': 0.116, 'prompt_length': 1067.21, 'perplexity': 1.161, 'retrieval_consistency': 0.07, 'syntax_error_percent': 0.108, 'semantic_error_percent': 0.427},
        "irrelevant_dummy": {'pass@1': 0.155, 'prompt_length': 2409.949, 'perplexity': 1.156, 'syntax_error_percent': 0.045, 'semantic_error_percent': 0.395},
        "irrelevant_diff": {'pass@1': 0.158, 'prompt_length': 2409.949, 'perplexity': 1.154, 'syntax_error_percent': 0.07, 'semantic_error_percent': 0.369},
        "none": {'pass@1': 0.166, 'prompt_length': 514.739, 'perplexity': 1.15, 'syntax_error_percent': 0.032, 'semantic_error_percent': 0.401},
    },
    "pandas_numpy_eval": {
        "oracle": {'pass@1': 0.599, 'prompt_length': 1223.281, 'perplexity': 1.124, 'retrieval_consistency': 0.964, 'syntax_error_percent': 0.054, 'semantic_error_percent': 0.198},
        # "retrieved": {'pass@1': 0.38922155688622756},
        "distracting": {'pass@1': 0.491, 'prompt_length': 691.868, 'perplexity': 1.132, 'retrieval_consistency': 0.395, 'syntax_error_percent': 0.078, 'semantic_error_percent': 0.359},
        "random": {'pass@1': 0.617, 'prompt_length': 512.156, 'perplexity': 1.124, 'retrieval_consistency': 0.006, 'syntax_error_percent': 0.108, 'semantic_error_percent': 0.311},
        "irrelevant_dummy": {'pass@1': 0.617, 'prompt_length': 1318.581, 'perplexity': 1.134, 'syntax_error_percent': 0.078, 'semantic_error_percent': 0.323},
        "irrelevant_diff": {'pass@1': 0.671, 'prompt_length': 1318.581, 'perplexity': 1.11, 'syntax_error_percent': 0.036, 'semantic_error_percent': 0.287},
        "none": {'pass@1': 0.617, 'prompt_length': 187.036, 'perplexity': 1.098, 'syntax_error_percent': 0.09, 'semantic_error_percent': 0.347},
    }
}

code_ret_doc_type_gpt_n_1 = {
    "conala": {
        "oracle": {'pass@1': 0.381, 'prompt_length': 820.869, 'perplexity': 1.038, 'retrieval_consistency': 0.929, 'syntax_error_percent': 0.155, 'semantic_error_percent': 0.25},
        # "retrieved": {'pass@1': 0.21428571428571427},
        "distracting": {'pass@1': 0.226, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 178.619, 'perplexity': 1.046, 'retrieval_consistency': 0.452, 'syntax_error_percent': 0.286, 'semantic_error_percent': 0.452},
        "random": {'pass@1': 0.262, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 402.667, 'perplexity': 1.061, 'retrieval_consistency': 0.024, 'syntax_error_percent': 0.226, 'semantic_error_percent': 0.381},
        "irrelevant_dummy": {'pass@1': 0.31, 'prompt_length': 814.321, 'perplexity': 1.048, 'syntax_error_percent': 0.321, 'semantic_error_percent': 0.417},
        "irrelevant_diff": {'pass@1': 0.333, 'prompt_length': 814.833, 'perplexity': 1.047, 'syntax_error_percent': 0.321, 'semantic_error_percent': 0.429},
        "none": {'pass@1': 0.226, 'prompt_length': 66.369, 'perplexity': 1.046, 'syntax_error_percent': 0.417, 'semantic_error_percent': 0.393}
    },
    "DS1000": {
        "oracle": {'pass@1': 0.356, 'prompt_length': 2172.236, 'perplexity': 1.033, 'retrieval_consistency': 1.287, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.191},
        # "retrieved": {'pass@1': 0.25234407087954},
        "distracting": {'pass@1': 0.264, 'prompt_length': 1898.764, 'perplexity': 1.037, 'retrieval_consistency': 0.573, 'syntax_error_percent': 0.019, 'semantic_error_percent': 0.363},
        "random": {'pass@1': 0.351, 'prompt_length': 939.217, 'perplexity': 1.038, 'retrieval_consistency': 0.0, 'syntax_error_percent': 0.019, 'semantic_error_percent': 0.401},
        "irrelevant_dummy": {'pass@1': 0.356, 'prompt_length': 2192.28, 'perplexity': 1.037, 'syntax_error_percent': 0.019, 'semantic_error_percent': 0.344},
        "irrelevant_diff": {'pass@1': 0.399, 'prompt_length': 2193.631, 'perplexity': 1.039, 'syntax_error_percent': 0.019, 'semantic_error_percent': 0.318},
        "none": {'pass@1': 0.367, 'prompt_length': 413.535, 'perplexity': 1.044, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.312},
    },
    "pandas_numpy_eval": {
        "oracle": {'pass@1': 0.784, 'prompt_length': 1164.731, 'perplexity': 1.026, 'retrieval_consistency': 1.114, 'syntax_error_percent': 0.036, 'semantic_error_percent': 0.126},
        # "retrieved": {'pass@1': 0.6586826347305389},
        "distracting": {'pass@1': 0.653, 'prompt_length': 619.629, 'perplexity': 1.037, 'retrieval_consistency': 0.365, 'syntax_error_percent': 0.024, 'semantic_error_percent': 0.347},
        "random": {'pass@1': 0.754, 'prompt_length': 449.988, 'perplexity': 1.032, 'retrieval_consistency': 0.006, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.275},
        "irrelevant_dummy": {'pass@1': 0.772, 'prompt_length': 1190.766, 'perplexity': 1.026, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.293},
        "irrelevant_diff": {'pass@1': 0.79, 'prompt_length': 1191.898, 'perplexity': 1.026, 'syntax_error_percent': 0.024, 'semantic_error_percent': 0.293},
        "none": {'pass@1': 0.79, 'prompt_length': 142.186, 'perplexity': 1.028, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.281}
    }
}


qa_ret_recall_llama_n_1 = {
    'NQ': {
        1: {'em': 0.203, 'f1': 0.398, 'prec': 0.345, 'recall': 0.806, 'has_answer': 0.735, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 283.926, 'perplexity': 1.061},
        0.8: {'em': 0.171, 'f1': 0.347, 'prec': 0.298, 'recall': 0.716, 'has_answer': 0.635, 'ret_recall': 0.8, 'oracle_percent': 0.8, 'oracle_rank': 1.0, 'prompt_length': 283.483, 'perplexity': 1.064},
        0.6: {'em': 0.133, 'f1': 0.29, 'prec': 0.246, 'recall': 0.617, 'has_answer': 0.527, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 1.0, 'prompt_length': 282.443, 'perplexity': 1.066},
        0.4: {'em': 0.097, 'f1': 0.235, 'prec': 0.197, 'recall': 0.521, 'has_answer': 0.421, 'ret_recall': 0.4, 'oracle_percent': 0.4, 'oracle_rank': 1.0, 'prompt_length': 281.589, 'perplexity': 1.068},
        0.2: {'em': 0.067, 'f1': 0.185, 'prec': 0.153, 'recall': 0.427, 'has_answer': 0.322, 'ret_recall': 0.2, 'oracle_percent': 0.2, 'oracle_rank': 1.0, 'prompt_length': 281.065, 'perplexity': 1.071},
        0: {'em': 0.028, 'f1': 0.125, 'prec': 0.1, 'recall': 0.323, 'has_answer': 0.213, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 280.384, 'perplexity': 1.074},
    },
    "TriviaQA": {
        1.0: {'em': 0.358, 'f1': 0.523, 'prec': 0.461, 'recall': 0.904, 'has_answer': 0.888, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 295.269, 'perplexity': 1.062},
        0.8: {'em': 0.314, 'f1': 0.47, 'prec': 0.412, 'recall': 0.836, 'has_answer': 0.811, 'ret_recall': 0.8, 'oracle_percent': 0.8, 'oracle_rank': 1.0, 'prompt_length': 294.865, 'perplexity': 1.063},
        0.6: {'em': 0.263, 'f1': 0.416, 'prec': 0.361, 'recall': 0.771, 'has_answer': 0.734, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 1.0, 'prompt_length': 294.474, 'perplexity': 1.065},
        0.4: {'em': 0.217, 'f1': 0.37, 'prec': 0.317, 'recall': 0.723, 'has_answer': 0.67, 'ret_recall': 0.4, 'oracle_percent': 0.4, 'oracle_rank': 1.0, 'prompt_length': 293.97, 'perplexity': 1.066},
        0.2: {'em': 0.166, 'f1': 0.318, 'prec': 0.265, 'recall': 0.667, 'has_answer': 0.6, 'ret_recall': 0.2, 'oracle_percent': 0.2, 'oracle_rank': 1.0, 'prompt_length': 293.853, 'perplexity': 1.068},
        0.0: {'em': 0.115, 'f1': 0.264, 'prec': 0.212, 'recall': 0.61, 'has_answer': 0.528, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 293.599, 'perplexity': 1.069},
    },
    "hotpotQA": {
        1.0: {'em': 0.234, 'f1': 0.427, 'prec': 0.384, 'recall': 0.79, 'has_answer': 0.692, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.5, 'prompt_length': 371.815, 'perplexity': 1.057},
        0.8: {'em': 0.197, 'f1': 0.37, 'prec': 0.332, 'recall': 0.695, 'has_answer': 0.598, 'ret_recall': 0.8, 'oracle_percent': 0.8, 'oracle_rank': 1.495, 'prompt_length': 365.962, 'perplexity': 1.059},
        0.6: {'em': 0.157, 'f1': 0.312, 'prec': 0.275, 'recall': 0.608, 'has_answer': 0.506, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 1.492, 'prompt_length': 358.126, 'perplexity': 1.061},
        0.4: {'em': 0.121, 'f1': 0.252, 'prec': 0.218, 'recall': 0.512, 'has_answer': 0.418, 'ret_recall': 0.4, 'oracle_percent': 0.4, 'oracle_rank': 1.486, 'prompt_length': 351.218, 'perplexity': 1.064},
        0.2: {'em': 0.084, 'f1': 0.191, 'prec': 0.163, 'recall': 0.404, 'has_answer': 0.31, 'ret_recall': 0.2, 'oracle_percent': 0.2, 'oracle_rank': 1.458, 'prompt_length': 342.573, 'perplexity': 1.068},
        0.0: {'em': 0.054, 'f1': 0.134, 'prec': 0.113, 'recall': 0.293, 'has_answer': 0.201, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 334.702, 'perplexity': 1.071},
    }
}


qa_ret_recall_gpt_n_1 = {
    'NQ': {
        1.0: {'em': 0.523, 'f1': 0.681, 'prec': 0.663, 'recall': 0.822, 'has_answer': 0.77, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 228.052, 'perplexity': 1.027},
        0.8: {'em': 0.434, 'f1': 0.585, 'prec': 0.573, 'recall': 0.709, 'has_answer': 0.648, 'ret_recall': 0.8, 'oracle_percent': 0.8, 'oracle_rank': 1.0, 'prompt_length': 227.672, 'perplexity': 1.032},
        0.6: {'em': 0.345, 'f1': 0.487, 'prec': 0.48, 'recall': 0.596, 'has_answer': 0.526, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 1.0, 'prompt_length': 227.066, 'perplexity': 1.035},
        0.4: {'em': 0.258, 'f1': 0.391, 'prec': 0.39, 'recall': 0.484, 'has_answer': 0.401, 'ret_recall': 0.4, 'oracle_percent': 0.4, 'oracle_rank': 1.0, 'prompt_length': 226.524, 'perplexity': 1.041},
        0.2: {'em': 0.165, 'f1': 0.294, 'prec': 0.3, 'recall': 0.371, 'has_answer': 0.275, 'ret_recall': 0.2, 'oracle_percent': 0.2, 'oracle_rank': 1.0, 'prompt_length': 226.031, 'perplexity': 1.046},
        0.0: {'em': 0.077, 'f1': 0.192, 'prec': 0.198, 'recall': 0.262, 'has_answer': 0.158, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 225.648, 'perplexity': 1.05}
    },
    "TriviaQA": {
        1.0: {'em': 0.729, 'f1': 0.808, 'prec': 0.783, 'recall': 0.895, 'has_answer': 0.888, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 237.544, 'perplexity': 1.018},
        0.8: {'em': 0.666, 'f1': 0.743, 'prec': 0.72, 'recall': 0.827, 'has_answer': 0.813, 'ret_recall': 0.8, 'oracle_percent': 0.8, 'oracle_rank': 1.0, 'prompt_length': 237.381, 'perplexity': 1.024},
        0.6: {'em': 0.596, 'f1': 0.677, 'prec': 0.66, 'recall': 0.757, 'has_answer': 0.731, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 1.0, 'prompt_length': 237.06, 'perplexity': 1.028},
        0.4: {'em': 0.532, 'f1': 0.614, 'prec': 0.6, 'recall': 0.691, 'has_answer': 0.654, 'ret_recall': 0.4, 'oracle_percent': 0.4, 'oracle_rank': 1.0, 'prompt_length': 236.735, 'perplexity': 1.033},
        0.2: {'em': 0.458, 'f1': 0.544, 'prec': 0.532, 'recall': 0.623, 'has_answer': 0.58, 'ret_recall': 0.2, 'oracle_percent': 0.2, 'oracle_rank': 1.0, 'prompt_length': 236.626, 'perplexity': 1.037},
        0.0: {'em': 0.399, 'f1': 0.488, 'prec': 0.477, 'recall': 0.569, 'has_answer': 0.514, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 236.456, 'perplexity': 1.04}
    },
    "hotpotQA": {
        1.0: {'em': 0.528, 'f1': 0.693, 'prec': 0.708, 'recall': 0.75, 'has_answer': 0.684, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.5, 'prompt_length': 300.92, 'perplexity': 1.021},
        0.8: {'em': 0.453, 'f1': 0.604, 'prec': 0.617, 'recall': 0.659, 'has_answer': 0.585, 'ret_recall': 0.8, 'oracle_percent': 0.8, 'oracle_rank': 1.495, 'prompt_length': 296.255, 'perplexity': 1.027},
        0.6: {'em': 0.372, 'f1': 0.514, 'prec': 0.528, 'recall': 0.562, 'has_answer': 0.492, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 1.492, 'prompt_length': 289.354, 'perplexity': 1.033},
        0.4: {'em': 0.309, 'f1': 0.428, 'prec': 0.436, 'recall': 0.475, 'has_answer': 0.417, 'ret_recall': 0.4, 'oracle_percent': 0.4, 'oracle_rank': 1.486, 'prompt_length': 283.193, 'perplexity': 1.039},
        0.2: {'em': 0.234, 'f1': 0.337, 'prec': 0.345, 'recall': 0.376, 'has_answer': 0.316, 'ret_recall': 0.2, 'oracle_percent': 0.2, 'oracle_rank': 1.458, 'prompt_length': 276.349, 'perplexity': 1.045},
        0.0: {'em': 0.145, 'f1': 0.232, 'prec': 0.235, 'recall': 0.268, 'has_answer': 0.213, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 269.839, 'perplexity': 1.053},
    }
}

qa_ret_doc_type_llama_n_1 = {
    'NQ': {
        "oracle": {'em': 0.203, 'f1': 0.398, 'prec': 0.345, 'recall': 0.806, 'has_answer': 0.735, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 283.926, 'perplexity': 1.061},
        # "retrieved": {'em': 0.1095, 'f1': 0.24555840009200883, 'prec': 0.20931816465800238, 'recall': 0.5241250000000001},
        "distracting": {'em': 0.028, 'f1': 0.125, 'prec': 0.1, 'recall': 0.323, 'has_answer': 0.213, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 280.384, 'perplexity': 1.074},
        "random": {'em': 0.002, 'f1': 0.073, 'prec': 0.046, 'recall': 0.281, 'has_answer': 0.211, 'ret_recall': 0.003, 'oracle_percent': 0.003, 'oracle_rank': 1.0, 'prompt_length': 284.834, 'perplexity': 1.11},
        "irrelevant_dummy": {'em': 0.005, 'f1': 0.08, 'prec': 0.052, 'recall': 0.268, 'has_answer': 0.198, 'prompt_length': 283.926, 'perplexity': 1.105},
        "irrelevant_diff": {'em': 0.003, 'f1': 0.105, 'prec': 0.065, 'recall': 0.379, 'has_answer': 0.293, 'prompt_length': 283.923, 'perplexity': 1.12},
        "none": {'em': 0.013, 'f1': 0.121, 'prec': 0.078, 'recall': 0.509, 'has_answer': 0.401, 'prompt_length': 89.734, 'perplexity': 1.087},
    },
    "TriviaQA": {
        "oracle": {'em': 0.358, 'f1': 0.523, 'prec': 0.461, 'recall': 0.904, 'has_answer': 0.888, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 295.269, 'perplexity': 1.062},
        # "retrieved": {'em': 0.298, 'f1': 0.44355579510359733, 'prec': 0.39018332836747666, 'recall': 0.7751964285714283},
        "distracting": {'em': 0.115, 'f1': 0.264, 'prec': 0.212, 'recall': 0.61, 'has_answer': 0.528, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 293.599, 'perplexity': 1.069},
        "random": {'em': 0.048, 'f1': 0.223, 'prec': 0.157, 'recall': 0.668, 'has_answer': 0.611, 'ret_recall': 0.006, 'oracle_percent': 0.006, 'oracle_rank': 1.0, 'prompt_length': 294.918, 'perplexity': 1.087},
        "irrelevant_dummy": {'em': 0.089, 'f1': 0.271, 'prec': 0.205, 'recall': 0.69, 'has_answer': 0.635, 'prompt_length': 295.269, 'perplexity': 1.081},
        "irrelevant_diff": {'em': 0.06, 'f1': 0.239, 'prec': 0.17, 'recall': 0.717, 'has_answer': 0.66, 'prompt_length': 295.264, 'perplexity': 1.097},
        "none": {'em': 0.104, 'f1': 0.279, 'prec': 0.21, 'recall': 0.791, 'has_answer': 0.737, 'prompt_length': 99.819, 'perplexity': 1.06}
    },
    "hotpotQA": {
        "oracle": {'em': 0.235, 'f1': 0.426, 'prec': 0.385, 'recall': 0.781, 'has_answer': 0.684, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.5, 'prompt_length': 371.815, 'perplexity': 1.057},
        # "retrieved": {'em': 0.1135, 'f1': 0.23571233027433397, 'prec': 0.20502697902833195, 'recall': 0.47636953046953057},
        "distracting": {'em': 0.059, 'f1': 0.136, 'prec': 0.115, 'recall': 0.291, 'has_answer': 0.198, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 334.702, 'perplexity': 1.07},
        "random": {'em': 0.037, 'f1': 0.12, 'prec': 0.093, 'recall': 0.293, 'has_answer': 0.206, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 287.745, 'perplexity': 1.085},
        "irrelevant_dummy": {'em': 0.053, 'f1': 0.139, 'prec': 0.113, 'recall': 0.304, 'has_answer': 0.21, 'prompt_length': 371.815, 'perplexity': 1.092},
        "irrelevant_diff": {'em': 0.035, 'f1': 0.122, 'prec': 0.094, 'recall': 0.312, 'has_answer': 0.226, 'prompt_length': 371.811, 'perplexity': 1.105},
        "none": {'em': 0.022, 'f1': 0.115, 'prec': 0.083, 'recall': 0.36, 'has_answer': 0.277, 'prompt_length': 101.672, 'perplexity': 1.073},
    }
}

qa_ret_doc_type_gpt_n_1 = {
    'NQ': {
        "oracle": {'em': 0.519, 'f1': 0.676, 'prec': 0.657, 'recall': 0.82, 'has_answer': 0.77, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 228.052, 'perplexity': 1.027},
        "retrieved": {'em': 0.251, 'f1': 0.39, 'prec': 0.38, 'recall': 0.501, 'ret_recall': 0.472, 'oracle_percent': 0.472, 'oracle_rank': 1.0, 'prompt_length': 226.675, 'perplexity': 1.039},
        "distracting": {'em': 0.077, 'f1': 0.189, 'prec': 0.195, 'recall': 0.257, 'has_answer': 0.154, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 225.648, 'perplexity': 1.05},
        "random": {'em': 0.164, 'f1': 0.265, 'prec': 0.263, 'recall': 0.355, 'has_answer': 0.279, 'ret_recall': 0.003, 'oracle_percent': 0.003, 'oracle_rank': 1.0, 'prompt_length': 230.421, 'perplexity': 1.077},
        "irrelevant_dummy": {'em': 0.12, 'f1': 0.198, 'prec': 0.188, 'recall': 0.288, 'has_answer': 0.22, 'prompt_length': 228.262, 'perplexity': 1.087},
        "irrelevant_diff": {'em': 0.142, 'f1': 0.227, 'prec': 0.224, 'recall': 0.307, 'has_answer': 0.235, 'prompt_length': 228.579, 'perplexity': 1.088},
        "none": {'em': 0.247, 'f1': 0.403, 'prec': 0.381, 'recall': 0.603, 'has_answer': 0.496, 'prompt_length': 64.483, 'perplexity': 1.064},
        "ellipsis": {'em': 0.257, 'f1': 0.386, 'prec': 0.382, 'recall': 0.505, 'has_answer': 0.408, 'prompt_length': 229.344, 'perplexity': 1.071},
    },
    "TriviaQA": {
        "oracle": {'em': 0.734, 'f1': 0.812, 'prec': 0.786, 'recall': 0.898, 'has_answer': 0.892, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 237.544, 'perplexity': 1.018},
        # "retrieved": {'em': 0.588, 'f1': 0.6785403596045625, 'prec': 0.6573662049086141, 'recall': 0.7640666666666663},
        "distracting": {'em': 0.401, 'f1': 0.489, 'prec': 0.478, 'recall': 0.569, 'has_answer': 0.516, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 236.456, 'perplexity': 1.04},
        "random": {'em': 0.65, 'f1': 0.716, 'prec': 0.697, 'recall': 0.791, 'has_answer': 0.766, 'ret_recall': 0.006, 'oracle_percent': 0.006, 'oracle_rank': 1.0, 'prompt_length': 237.699, 'perplexity': 1.04},
        "irrelevant_dummy": {'em': 0.645, 'f1': 0.714, 'prec': 0.693, 'recall': 0.807, 'has_answer': 0.779, 'prompt_length': 237.794, 'perplexity': 1.042},
        "irrelevant_diff": {'em': 0.666, 'f1': 0.728, 'prec': 0.712, 'recall': 0.797, 'has_answer': 0.771, 'prompt_length': 238.488, 'perplexity': 1.043},
        "none": {'em': 0.706, 'f1': 0.773, 'prec': 0.748, 'recall': 0.878, 'has_answer': 0.862, 'prompt_length': 71.761, 'perplexity': 1.029},
    },
    "hotpotQA": {
        "oracle": {'em': 0.521, 'f1': 0.688, 'prec': 0.702, 'recall': 0.745, 'has_answer': 0.676, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.5, 'prompt_length': 300.921, 'perplexity': 1.021},
        # "retrieved": {'em': 0.2615, 'f1': 0.38276433036531554, 'prec': 0.38988646984133074, 'recall': 0.43652234154734176},
        "distracting": {'em': 0.147, 'f1': 0.234, 'prec': 0.238, 'recall': 0.269, 'has_answer': 0.209, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 269.839, 'perplexity': 1.055},
        "random": {'em': 0.175, 'f1': 0.26, 'prec': 0.262, 'recall': 0.301, 'has_answer': 0.227, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 231.194, 'perplexity': 1.08},
        "irrelevant_dummy": {'em': 0.183, 'f1': 0.263, 'prec': 0.267, 'recall': 0.311, 'has_answer': 0.233, 'prompt_length': 303.166, 'perplexity': 1.087},
        "irrelevant_diff": {'em': 0.196, 'f1': 0.276, 'prec': 0.285, 'recall': 0.31, 'has_answer': 0.231, 'prompt_length': 304.324, 'perplexity': 1.091},
        "none": {'em': 0.202, 'f1': 0.333, 'prec': 0.336, 'recall': 0.391, 'has_answer': 0.324, 'prompt_length': 73.689, 'perplexity': 1.084},
    }
}



# todo: top1 top5 top10 top15 top20 for code llama (16k) and gpt (16k)
code_ret_doc_selection_topk_llama_n_1 = {
    'conala': {
        'top_1': {'pass@1': 0.167, 'prompt_length': 192.631, 'perplexity': 1.133, 'retrieval_consistency': 0.321, 'syntax_error_percent': 0.179, 'semantic_error_percent': 0.5},
        'top_5': {'pass@1': 0.226, 'prompt_length': 542.631, 'perplexity': 1.129, 'retrieval_consistency': 0.738, 'syntax_error_percent': 0.226, 'semantic_error_percent': 0.44},
        'top_10': {'pass@1': 0.214, 'prompt_length': 968.214, 'perplexity': 1.131, 'retrieval_consistency': 1.214, 'syntax_error_percent': 0.226, 'semantic_error_percent': 0.464},
        'top_15': {'pass@1': 0.19, 'prompt_length': 1538.417, 'perplexity': 1.132, 'retrieval_consistency': 1.476, 'syntax_error_percent': 0.274, 'semantic_error_percent': 0.429},
        'top_20': {'pass@1': 0.155, 'prompt_length': 2045.929, 'perplexity': 1.129, 'retrieval_consistency': 1.607, 'syntax_error_percent': 0.274, 'semantic_error_percent': 0.44}
    },
    'DS1000': {
        'top_1': {'pass@1': 0.149, 'prompt_length': 1155.414, 'perplexity': 1.155, 'retrieval_consistency': 0.401, 'syntax_error_percent': 0.096, 'semantic_error_percent': 0.42},
        'top_5': {'pass@1': 0.095, 'prompt_length': 3275.248, 'perplexity': 1.138, 'retrieval_consistency': 1.255, 'syntax_error_percent': 0.134, 'semantic_error_percent': 0.401},
        'top_10': {'pass@1': 0.092, 'prompt_length': 5470.917, 'perplexity': 1.132, 'retrieval_consistency': 1.777, 'syntax_error_percent': 0.172, 'semantic_error_percent': 0.433},
        'top_15': {'pass@1': 0.138, 'prompt_length': 6466.204, 'perplexity': 1.126, 'retrieval_consistency': 2.172, 'syntax_error_percent': 0.197, 'semantic_error_percent': 0.433},
        'top_20': {'pass@1': 0.089, 'prompt_length': 6848.389, 'perplexity': 1.124, 'retrieval_consistency': 2.662, 'syntax_error_percent': 0.236, 'semantic_error_percent': 0.427}
    },
    'pandas_numpy_eval': {
        'top_1': {'pass@1': 0.527, 'prompt_length': 573.844, 'perplexity': 1.126, 'retrieval_consistency': 0.341, 'syntax_error_percent': 0.084, 'semantic_error_percent': 0.359},
        'top_5': {'pass@1': 0.551, 'prompt_length': 1935.874, 'perplexity': 1.135, 'retrieval_consistency': 0.964, 'syntax_error_percent': 0.042, 'semantic_error_percent': 0.323},
        'top_10': {'pass@1': 0.599, 'prompt_length': 3660.982, 'perplexity': 1.144, 'retrieval_consistency': 1.347, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.335},
        'top_15': {'pass@1': 0.629, 'prompt_length': 5122.91, 'perplexity': 1.137, 'retrieval_consistency': 1.653, 'syntax_error_percent': 0.042, 'semantic_error_percent': 0.287},
        'top_20': {'pass@1': 0.581, 'prompt_length': 6155.419, 'perplexity': 1.143, 'retrieval_consistency': 1.928, 'syntax_error_percent': 0.054, 'semantic_error_percent': 0.305}
    }
}

# code_ret_doc_selection_gpt_n_1 = {
#     'conala': {
#         "top_1": {'pass@1': 0.17857142857142858},
#         "top_3": {'pass@1': 0.32142857142857145},
#         "top_5": {'pass@1': 0.27380952380952384},
#         "top_7": {'pass@1': 0.2619047619047619},
#         "top_9": {'pass@1': 0.27380952380952384},
#     },
#     'DS1000': {
#         "top_1": {'pass@1': 0.2352011578556201},
#         "top_3": {'pass@1': 0.3392282266424143},
#         "top_5": {'pass@1': 0.3233258033601283},
#         "top_7": {'pass@1': 0.3295707915273133},
#         "top_9": {'pass@1': 0.3460428523929668},
#     },
#     'pandas_numpy_eval': {
#         "top_1": {'pass@1': 0.6287425149700598},
#         "top_3": {'pass@1': 0.688622754491018},
#         "top_5": {'pass@1': 0.7245508982035929},
#         "top_7": {'pass@1': 0.7305389221556886},
#         "top_9": {'pass@1': 0.7305389221556886},
#     }
# }


code_ret_doc_selection_topk_gpt_n_1 = {
    'conala': {
        'top_1': {'pass@1': 0.179, 'prompt_length': 157.976, 'perplexity': 1.049, 'retrieval_consistency': 0.381, 'syntax_error_percent': 0.357, 'semantic_error_percent': 0.452, 'ret_recall': 0.004},
        'top_5': {'pass@1': 0.286, 'prompt_length': 484.774, 'perplexity': 1.046, 'retrieval_consistency': 0.81, 'syntax_error_percent': 0.202, 'semantic_error_percent': 0.405, 'ret_recall': 0.099},
        'top_10': {'pass@1': 0.286, 'prompt_length': 875.19, 'perplexity': 1.043, 'retrieval_consistency': 1.333, 'syntax_error_percent': 0.25, 'semantic_error_percent': 0.381, 'ret_recall': 0.151},
        'top_15': {'pass@1': 0.25, 'prompt_length': 1405.833, 'perplexity': 1.044, 'retrieval_consistency': 1.583, 'syntax_error_percent': 0.262, 'semantic_error_percent': 0.405, 'ret_recall': 0.193},
        'top_20': {'pass@1': 0.25, 'prompt_length': 1897.905, 'perplexity': 1.047, 'retrieval_consistency': 1.881, 'syntax_error_percent': 0.25, 'semantic_error_percent': 0.393, 'ret_recall': 0.211}
    },
    'DS1000': {
        'top_1': {'pass@1': 0.238, 'prompt_length': 1045.637, 'perplexity': 1.044, 'retrieval_consistency': 0.376, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.427, 'ret_recall': 0.043},
        'top_5': {'pass@1': 0.339, 'prompt_length': 3139.516, 'perplexity': 1.039, 'retrieval_consistency': 0.911, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.369, 'ret_recall': 0.117},
        'top_10': {'pass@1': 0.362, 'prompt_length': 5508.363, 'perplexity': 1.039, 'retrieval_consistency': 1.611, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.325, 'ret_recall': 0.181},
        'top_15': {'pass@1': 0.348, 'prompt_length': 6281.465, 'perplexity': 1.043, 'retrieval_consistency': 1.885, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.338, 'ret_recall': 0.21},
        'top_20': {'pass@1': 0.342, 'prompt_length': 6745.025, 'perplexity': 1.047, 'retrieval_consistency': 2.159, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.299, 'ret_recall': 0.229}
    },
    'pandas_numpy_eval': {
        'top_1': {'pass@1': 0.635, 'prompt_length': 507.605, 'perplexity': 1.038, 'retrieval_consistency': 0.377, 'syntax_error_percent': 0.024, 'semantic_error_percent': 0.347, 'ret_recall': 0.099},
        'top_5': {'pass@1': 0.719, 'prompt_length': 1805.76, 'perplexity': 1.037, 'retrieval_consistency': 0.928, 'syntax_error_percent': 0.036, 'semantic_error_percent': 0.287, 'ret_recall': 0.222},
        'top_10': {'pass@1': 0.731, 'prompt_length': 3438.928, 'perplexity': 1.039, 'retrieval_consistency': 1.353, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.281, 'ret_recall': 0.294},
        'top_15': {'pass@1': 0.695, 'prompt_length': 4953.91, 'perplexity': 1.037, 'retrieval_consistency': 1.647, 'syntax_error_percent': 0.024, 'semantic_error_percent': 0.263, 'ret_recall': 0.357},
        'top_20': {'pass@1': 0.713, 'prompt_length': 6499.341, 'perplexity': 1.039, 'retrieval_consistency': 1.874, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.269, 'ret_recall': 0.395}
    }
}


qa_ret_doc_selection_topk_llama_n_1 = {
    'NQ': {
        "top_1": {'em': 0.11, 'f1': 0.246, 'prec': 0.209, 'recall': 0.524, 'has_answer': 0.435, 'ret_recall': 0.472, 'oracle_percent': 0.472, 'oracle_rank': 1.0, 'prompt_length': 282.027, 'perplexity': 1.068},
        "top_5": {'em': 0.04, 'f1': 0.19, 'prec': 0.136, 'recall': 0.64, 'has_answer': 0.544, 'ret_recall': 0.735, 'oracle_percent': 0.306, 'oracle_rank': 1.687, 'prompt_length': 921.669, 'perplexity': 1.068},
        "top_10": {'em': 0.064, 'f1': 0.211, 'prec': 0.16, 'recall': 0.657, 'has_answer': 0.559, 'ret_recall': 0.816, 'oracle_percent': 0.242, 'oracle_rank': 2.277, 'prompt_length': 1720.924, 'perplexity': 1.076},
        "top_15": {'em': 0.061, 'f1': 0.209, 'prec': 0.159, 'recall': 0.641, 'has_answer': 0.552, 'ret_recall': 0.847, 'oracle_percent': 0.21, 'oracle_rank': 2.666, 'prompt_length': 2527.202, 'perplexity': 1.076},
        "top_20": {'em': 0.011, 'f1': 0.104, 'prec': 0.068, 'recall': 0.567, 'has_answer': 0.546, 'ret_recall': 0.871, 'oracle_percent': 0.19, 'oracle_rank': 3.075, 'prompt_length': 3332.663, 'perplexity': 1.083},
    },
    "TriviaQA": {
        "top_1": {'em': 0.298, 'f1': 0.444, 'prec': 0.39, 'recall': 0.775, 'has_answer': 0.732, 'ret_recall': 0.627, 'oracle_percent': 0.627, 'oracle_rank': 1.0, 'prompt_length': 293.966, 'perplexity': 1.063},
        "top_5": {'em': 0.146, 'f1': 0.331, 'prec': 0.259, 'recall': 0.838, 'has_answer': 0.801, 'ret_recall': 0.871, 'oracle_percent': 0.511, 'oracle_rank': 1.5, 'prompt_length': 946.482, 'perplexity': 1.06},
        "top_10": {'em': 0.202, 'f1': 0.378, 'prec': 0.309, 'recall': 0.861, 'has_answer': 0.825, 'ret_recall': 0.917, 'oracle_percent': 0.443, 'oracle_rank': 1.8, 'prompt_length': 1759.983, 'perplexity': 1.068},
        "top_15": {'em': 0.127, 'f1': 0.314, 'prec': 0.241, 'recall': 0.84, 'has_answer': 0.836, 'ret_recall': 0.937, 'oracle_percent': 0.401, 'oracle_rank': 2.03, 'prompt_length': 2579.584, 'perplexity': 1.073},
        "top_20": {'em': 0.014, 'f1': 0.142, 'prec': 0.089, 'recall': 0.743, 'has_answer': 0.835, 'ret_recall': 0.952, 'oracle_percent': 0.375, 'oracle_rank': 2.271, 'prompt_length': 3398.936, 'perplexity': 1.084}
    },
    "hotpotQA": {
        "top_1": {'em': 0.123, 'f1': 0.242, 'prec': 0.216, 'recall': 0.457, 'has_answer': 0.354, 'ret_recall': 0.351, 'oracle_percent': 0.703, 'oracle_rank': 1.0, 'prompt_length': 247.024, 'perplexity': 1.062},
        "top_5": {'em': 0.077, 'f1': 0.199, 'prec': 0.159, 'recall': 0.521, 'has_answer': 0.415, 'ret_recall': 0.582, 'oracle_percent': 0.233, 'oracle_rank': 1.7, 'prompt_length': 656.491, 'perplexity': 1.06},
        "top_10": {'em': 0.046, 'f1': 0.169, 'prec': 0.125, 'recall': 0.536, 'has_answer': 0.443, 'ret_recall': 0.639, 'oracle_percent': 0.128, 'oracle_rank': 2.225, 'prompt_length': 1150.41, 'perplexity': 1.06},
        "top_15": {'em': 0.036, 'f1': 0.162, 'prec': 0.12, 'recall': 0.523, 'has_answer': 0.436, 'ret_recall': 0.666, 'oracle_percent': 0.089, 'oracle_rank': 2.665, 'prompt_length': 1648.619, 'perplexity': 1.065},
        "top_20": {'em': 0.03, 'f1': 0.145, 'prec': 0.105, 'recall': 0.498, 'has_answer': 0.429, 'ret_recall': 0.688, 'oracle_percent': 0.069, 'oracle_rank': 3.172, 'prompt_length': 2144.122, 'perplexity': 1.069}
    }
}

# qa_ret_doc_selection_gpt_n_1 = {
#     'NQ': {
#         "top_1": {'em': 0.25, 'f1': 0.3922751526271766, 'prec': 0.38121272333562434, 'recall': 0.5068083333333334},
#         "top_5": {'em': 0.3265, 'f1': 0.4678190562927842, 'prec': 0.4536730708690414, 'recall': 0.6030250000000003},
#         "top_10": {'em': 0.3445, 'f1': 0.49073557800297163, 'prec': 0.4763133890449864, 'recall': 0.6221250000000003},
#         "top_15": {'em': 0.345, 'f1': 0.49211529865522696, 'prec': 0.47809000351015996, 'recall': 0.6243750000000001},
#         "top_20": {'em': 0.349, 'f1': 0.4971804164263661, 'prec': 0.4826739429489419, 'recall': 0.6316583333333337},
#         "top_25": {'em': 0.3465, 'f1': 0.4960016340178994, 'prec': 0.48273941627714634, 'recall': 0.6302333333333335},
#         "top_30": {'em': 0.349, 'f1': 0.4989983533747123, 'prec': 0.4866316346933494, 'recall': 0.6319000000000004},
#     },
#     "TriviaQA": {
#         "top_1": {'em': 0.5915, 'f1': 0.6793697892492293, 'prec': 0.6579298292768786, 'recall': 0.7669583333333331},
#         "top_5": {'em': 0.645, 'f1': 0.7274278104337638, 'prec': 0.7058988717589391, 'recall': 0.8117416666666667},
#         "top_10": {'em': 0.6705, 'f1': 0.7523535426367305, 'prec': 0.7286757299584995, 'recall': 0.8371984848484849},
#         "top_15": {'em': 0.673, 'f1': 0.7568096785435148, 'prec': 0.7334095129282513, 'recall': 0.8432704545454547},
#         "top_20": {'em': 0.677, 'f1': 0.7580148586678289, 'prec': 0.7353244827862068, 'recall': 0.8397083333333334},
#         "top_25": {'em': 0.692, 'f1': 0.773757922403883, 'prec': 0.7502785240860561, 'recall': 0.8545772727272728},
#         "top_30": {'em': 0.684, 'f1': 0.7655187413543476, 'prec': 0.7423013409859317, 'recall': 0.8463787878787878},
#     },
#     "hotpotQA": {
#         "top_1": {'em': 0.234, 'f1': 0.3553180842318932, 'prec': 0.36114732204224864, 'recall': 0.40814816572316587},
#         "top_5": {'em': 0.2785, 'f1': 0.408006665352049, 'prec': 0.41574636258581127, 'recall': 0.46214478021978056},
#         "top_10": {'em': 0.295, 'f1': 0.42051786332163515, 'prec': 0.42574595857381387, 'recall': 0.47493446275946294},
#         "top_15": {'em': 0.305, 'f1': 0.4328039911798426, 'prec': 0.4391074937975743, 'recall': 0.4849971250971255},
#         "top_20": {'em': 0.3055, 'f1': 0.4314498604767209, 'prec': 0.4389742696871426, 'recall': 0.48317771117771147},
#         "top_25": {'em': 0.3045, 'f1': 0.4363582397053665, 'prec': 0.44220459519321953, 'recall': 0.4943199730824735},
#         "top_30": {'em': 0.314, 'f1': 0.44482039894541053, 'prec': 0.4523217160423505, 'recall': 0.4990813603063606},
#     }
# }

qa_ret_doc_selection_topk_gpt_n_1 = {
    'NQ': {
        'top_1': {'em': 0.254, 'f1': 0.391, 'prec': 0.381, 'recall': 0.5, 'has_answer': 0.427, 'ret_recall': 0.472, 'oracle_percent': 0.472, 'oracle_rank': 1.0, 'prompt_length': 226.675, 'perplexity': 1.04},
        "top_5": {'em': 0.327, 'f1': 0.468, 'prec': 0.454, 'recall': 0.603, 'has_answer': 0.525, 'prompt_length': 774.843, 'perplexity': 1.036, 'ret_recall': 0.735},
        "top_10": {'em': 0.345, 'f1': 0.491, 'prec': 0.476, 'recall': 0.622, 'has_answer': 0.543, 'prompt_length': 1460.296, 'perplexity': 1.039, 'ret_recall': 0.816},
        "top_15": {'em': 0.345, 'f1': 0.492, 'prec': 0.478, 'recall': 0.624, 'has_answer': 0.545, 'prompt_length': 2146.843, 'perplexity': 1.04, 'ret_recall': 0.847},
        'top_20': {'em': 0.349, 'f1': 0.497, 'prec': 0.483, 'recall': 0.629, 'has_answer': 0.545, 'ret_recall': 0.871, 'oracle_percent': 0.19, 'oracle_rank': 3.075, 'prompt_length': 2833.01, 'perplexity': 1.042},
        'top_40': {'em': 0.346, 'f1': 0.496, 'prec': 0.482, 'recall': 0.635, 'has_answer': 0.549, 'ret_recall': 0.911, 'oracle_percent': 0.147, 'oracle_rank': 4.238, 'prompt_length': 5577.404, 'perplexity': 1.048},
        'top_60': {'em': 0.344, 'f1': 0.492, 'prec': 0.476, 'recall': 0.638, 'has_answer': 0.556, 'ret_recall': 0.932, 'oracle_percent': 0.127, 'oracle_rank': 5.244, 'prompt_length': 8323.228, 'perplexity': 1.052},
        'top_80': {'em': 0.337, 'f1': 0.487, 'prec': 0.471, 'recall': 0.633, 'has_answer': 0.547, 'ret_recall': 0.944, 'oracle_percent': 0.114, 'oracle_rank': 6.049, 'prompt_length': 11070.615, 'perplexity': 1.06}
    },
    'TriviaQA': {
        'top_1': {'em': 0.592, 'f1': 0.679, 'prec': 0.658, 'recall': 0.767, 'has_answer': 0.74, 'ret_recall': 0.627, 'oracle_percent': 0.627, 'oracle_rank': 1.0, 'prompt_length': 236.653, 'perplexity': 1.028},
        "top_5": {'em': 0.645, 'f1': 0.727, 'prec': 0.706, 'recall': 0.812, 'has_answer': 0.789, 'prompt_length': 799.074, 'perplexity': 1.023, 'ret_recall': 0.871},
        "top_10": {'em': 0.671, 'f1': 0.752, 'prec': 0.729, 'recall': 0.837, 'has_answer': 0.818, 'prompt_length': 1500.321, 'perplexity': 1.023, 'ret_recall': 0.917},
        "top_15": {'em': 0.673, 'f1': 0.757, 'prec': 0.733, 'recall': 0.843, 'has_answer': 0.824, 'prompt_length': 2201.954, 'perplexity': 1.025, 'ret_recall': 0.937},
        'top_20': {'em': 0.677, 'f1': 0.758, 'prec': 0.735, 'recall': 0.84, 'has_answer': 0.82, 'ret_recall': 0.952, 'oracle_percent': 0.375, 'oracle_rank': 2.271, 'prompt_length': 2904.057, 'perplexity': 1.025},
        'top_40': {'em': 0.697, 'f1': 0.774, 'prec': 0.75, 'recall': 0.857, 'has_answer': 0.84, 'ret_recall': 0.975, 'oracle_percent': 0.316, 'oracle_rank': 2.883, 'prompt_length': 5714.368, 'perplexity': 1.029},
        'top_60': {'em': 0.7, 'f1': 0.775, 'prec': 0.752, 'recall': 0.856, 'has_answer': 0.841, 'ret_recall': 0.984, 'oracle_percent': 0.285, 'oracle_rank': 3.321, 'prompt_length': 8529.492, 'perplexity': 1.036},
        'top_80': {'em': 0.683, 'f1': 0.763, 'prec': 0.739, 'recall': 0.855, 'has_answer': 0.839, 'ret_recall': 0.986, 'oracle_percent': 0.263, 'oracle_rank': 3.459, 'prompt_length': 11346.743, 'perplexity': 1.043}
    },
    'hotpotQA': {
        'top_1': {'em': 0.234, 'f1': 0.355, 'prec': 0.361, 'recall': 0.408, 'has_answer': 0.346, 'ret_recall': 0.351, 'oracle_percent': 0.703, 'oracle_rank': 1.0, 'prompt_length': 195.152, 'perplexity': 1.041},
        "top_5": {'em': 0.2785, 'f1': 0.408, 'prec': 0.416, 'recall': 0.462, 'has_answer': 0.407, 'prompt_length': 543.775, 'perplexity': 1.039, 'ret_recall': 0.582},
        "top_10": {'em': 0.295, 'f1': 0.421, 'prec': 0.426, 'recall': 0.475, 'has_answer': 0.423, 'prompt_length': 963.867, 'perplexity': 1.039, 'ret_recall': 0.639},
        "top_15": {'em': 0.305, 'f1': 0.433, 'prec': 0.439, 'recall': 0.485, 'has_answer': 0.428, 'prompt_length': 1383.611, 'perplexity': 1.04, 'ret_recall': 0.666},
        'top_20': {'em': 0.305, 'f1': 0.431, 'prec': 0.439, 'recall': 0.483, 'has_answer': 0.427, 'ret_recall': 0.688, 'oracle_percent': 0.069, 'oracle_rank': 3.172, 'prompt_length': 1800.811, 'perplexity': 1.039},
        'top_40': {'em': 0.323, 'f1': 0.453, 'prec': 0.461, 'recall': 0.506, 'has_answer': 0.438, 'ret_recall': 0.732, 'oracle_percent': 0.037, 'oracle_rank': 4.764, 'prompt_length': 3459.514, 'perplexity': 1.043},
        'top_60': {'em': 0.311, 'f1': 0.442, 'prec': 0.45, 'recall': 0.494, 'has_answer': 0.429, 'ret_recall': 0.751, 'oracle_percent': 0.025, 'oracle_rank': 5.917, 'prompt_length': 5106.305, 'perplexity': 1.049},
        'top_80': {'em': 0.305, 'f1': 0.441, 'prec': 0.449, 'recall': 0.495, 'has_answer': 0.427, 'ret_recall': 0.77, 'oracle_percent': 0.019, 'oracle_rank': 7.506, 'prompt_length': 6743.218, 'perplexity': 1.054}
    }
}



# todo: doc selection by pl: 500 1000 1500 2000 for llama (4k); 500 2000 4000 6000 8000 for gpt (16k)

qa_ret_doc_selection_pl_gpt_n_1 = {
    'NQ': {},
    'TriviaQA': {},
    'hotpotQA': {}
}

qa_ret_doc_selection_pl_llama_n_1 = {
    'NQ': {},
    'TriviaQA': {},
    'hotpotQA': {}
}

# todo: pl 1000 2000 4000 6000 8000 for code llama and gpt

code_ret_doc_selection_pl_llama_n_1 = {
    'conala': {

    },
    'DS1000': {

    },
    'pandas_numpy_eval': {

    }
}

code_ret_doc_selection_pl_gpt_n_1 = {
    'conala': {

    },
    'DS1000': {

    },
    'pandas_numpy_eval': {

    }
}



# todo: prompt length analysis: pl=500 1000 1500 2000 for llama; 500 2000 4000 6000 8000 for gpt

# qa_pl_analysis_gpt_n_1 = {
#     'NQ': {
#         'oracle_top10': {'em': 0.533, 'f1': 0.6916228634702651, 'prec': 0.6729883714002561, 'recall': 0.8258000000000005},
#         'distracting_top10': {'em': 0.0995, 'f1': 0.22505139697448193, 'prec': 0.23412652439254966, 'recall': 0.28269999999999995},
#         'random_top10': {'em': 0.1855, 'f1': 0.30140874647701077, 'prec': 0.30451733094639355, 'recall': 0.3886083333333334},
#         'irrelevant_diff_top10': {'em': 0.1505, 'f1': 0.2516590830240432, 'prec': 0.24821379062303991, 'recall': 0.34620833333333334},
#         'irrelevant_dummy_top10': {'em': 0.145, 'f1': 0.24138619528558955, 'prec': 0.2311666719250147, 'recall': 0.34219166666666667}
#     },
#     'TriviaQA': {
#         "oracle_top10": {'em': 0.7405, 'f1': 0.8167501274111777, 'prec': 0.7911091913482974, 'recall': 0.9011666666666666},
#         "distracting_top10": {'em': 0.403, 'f1': 0.4909749859018985, 'prec': 0.48126900979027726, 'recall': 0.5581903966131908},
#         "random_top10": {'em': 0.671, 'f1': 0.7316760810304767, 'prec': 0.7153646208437773, 'recall': 0.7961125},
#         "irrelevant_diff_top10": {'em': 0.67, 'f1': 0.7297007214197764, 'prec': 0.7142052979297553, 'recall': 0.7971416666666666},
#         "irrelevant_dummy_top10": {'em': 0.6745, 'f1': 0.739580804760392, 'prec': 0.7209649724559586, 'recall': 0.8175746212121213},
#     },
#     'hotpotQA': {
#         "oracle_top10": {'em': 0.5425, 'f1': 0.7073541396923203, 'prec': 0.7199568933992317, 'recall': 0.7624801587301592},
#         "distracting_top10": {'em': 0.1535, 'f1': 0.24765849641844792, 'prec': 0.2535140550948018, 'recall': 0.27620553058053043},
#         "random_top10": {'em': 0.206, 'f1': 0.290075118871929, 'prec': 0.2967565277125306, 'recall': 0.32681951936951953},
#         "irrelevant_diff_top10": {'em': 0.202, 'f1': 0.2963647082209858, 'prec': 0.3060181072716487, 'recall': 0.3290111832611831},
#         "irrelevant_dummy_top10": {'em': 0.2215, 'f1': 0.31129415865827115, 'prec': 0.32212524737409665, 'recall': 0.34101713564213576},
#     },
# }

# "oracle": {'pass@1': 0.38095238095238093},
#         "retrieved": {'pass@1': 0.21428571428571427},
#         "distracting": {'pass@1': 0.2261904761904762},
#         "random": {'pass@1': 0.2619047619047619},
#         "irrelevant_dummy": {'pass@1': 0.30952380952380953},
#         "irrelevant_diff": {'pass@1': 0.3333333333333333},
#         "none": {'pass@1': 0.21428571428571427}


code_pl_analysis_gpt_n_1 = {
    'conala': {
        'oracle': {
            'oracle': {'pass@1': 0.381, 'prompt_length': 820.869, 'perplexity': 1.038, 'retrieval_consistency': 0.94, 'syntax_error_percent': 0.167, 'semantic_error_percent': 0.25},
            # 'oracle_500': {'pass@1': 0.333, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.226, 'prompt_length': 503.619},
            # 'oracle_2000': {'pass@1': 0.357, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.552, 'prompt_length': 2010.036},
            'oracle_repeat_4000': {'pass@1': 0.357, 'prompt_length': 3917.75, 'perplexity': 1.031, 'retrieval_consistency': 25.929, 'syntax_error_percent': 0.179, 'semantic_error_percent': 0.238},
            # 'oracle_8000': {'pass@1': 0.321, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.589, 'prompt_length': 7127.048}
            'oracle_pad_random_4000': {'pass@1': 0.393, 'prompt_length': 3960.631, 'perplexity': 1.036, 'retrieval_consistency': 26.81, 'syntax_error_percent': 0.167, 'semantic_error_percent': 0.214},
            'oracle_pad_repeat_random_4000': {'pass@1': 0.417, 'prompt_length': 3494.357, 'perplexity': 1.036, 'retrieval_consistency': 25.905, 'syntax_error_percent': 0.143, 'semantic_error_percent': 0.238},
            'oracle_pad_diff_4000': {'pass@1': 0.381, 'prompt_length': 3971.976, 'perplexity': 1.034, 'retrieval_consistency': 25.833, 'syntax_error_percent': 0.214, 'semantic_error_percent': 0.226},
            'oracle_pad_repeat_diff_4000': {'pass@1': 0.369, 'prompt_length': 3854.524, 'perplexity': 1.036, 'retrieval_consistency': 25.369, 'syntax_error_percent': 0.179, 'semantic_error_percent': 0.262},
            'oracle_pad_dummy_4000': {'pass@1': 0.381, 'prompt_length': 3980.667, 'perplexity': 1.041, 'retrieval_consistency': 26.821, 'syntax_error_percent': 0.155, 'semantic_error_percent': 0.238},
            'oracle_pad_ellipsis_4000': {'pass@1': 0.345, 'prompt_length': 3967.024, 'perplexity': 1.043, 'syntax_error_percent': 0.19, 'semantic_error_percent': 0.238}
        },
        'distracting': {
            'distracting': {'pass@1': 0.226, 'prompt_length': 178.619, 'perplexity': 1.046, 'retrieval_consistency': 0.452, 'syntax_error_percent': 0.286, 'semantic_error_percent': 0.452},
            'distracting_repeat_4000': {'pass@1': 0.274, 'prompt_length': 2970.179, 'perplexity': 1.041, 'retrieval_consistency': 21.857, 'syntax_error_percent': 0.298, 'semantic_error_percent': 0.417},
            'distracting_pad_random_4000': {'pass@1': 0.202, 'prompt_length': 2712.738, 'perplexity': 1.043, 'retrieval_consistency': 22.524, 'syntax_error_percent': 0.214, 'semantic_error_percent': 0.429},
            'distracting_pad_repeat_random_4000': {'pass@1': 0.179, 'prompt_length': 2562.262, 'perplexity': 1.042, 'retrieval_consistency': 28.857, 'syntax_error_percent': 0.202, 'semantic_error_percent': 0.429},
            'distracting_pad_diff_4000': {'pass@1': 0.25, 'prompt_length': 2976.464, 'perplexity': 1.053, 'retrieval_consistency': 28.976, 'syntax_error_percent': 0.298, 'semantic_error_percent': 0.405},
            'distracting_pad_repeat_diff_4000': {'pass@1': 0.202, 'prompt_length': 2882.81, 'perplexity': 1.048, 'retrieval_consistency': 29.655, 'syntax_error_percent': 0.333, 'semantic_error_percent': 0.345},
            'distracting_pad_dummy_4000': {'pass@1': 0.238, 'prompt_length': 3015.048, 'perplexity': 1.052, 'retrieval_consistency': 27.655, 'syntax_error_percent': 0.274, 'semantic_error_percent': 0.369},
            'distracting_pad_ellipsis_4000': {'pass@1': 0.179, 'prompt_length': 3011.952, 'perplexity': 1.057, 'syntax_error_percent': 0.369, 'semantic_error_percent': 0.381}
        },
        'retrieved_top': {
            'retrieved_top': {'pass@1': 0.286, 'prompt_length': 484.774, 'perplexity': 1.046, 'retrieval_consistency': 0.81, 'syntax_error_percent': 0.202, 'semantic_error_percent': 0.405},
            'retrieved_top_repeat_4000': {'pass@1': 0.298, 'prompt_length': 4001.869, 'perplexity': 1.046, 'retrieval_consistency': 26.095, 'syntax_error_percent': 0.202, 'semantic_error_percent': 0.357},
            'retrieved_top_pad_random_4000': {'pass@1': 0.262, 'prompt_length': 3650.155, 'perplexity': 1.041, 'retrieval_consistency': 24.417, 'syntax_error_percent': 0.202, 'semantic_error_percent': 0.452},
            'retrieved_top_pad_repeat_random_4000': {'pass@1': 0.262, 'prompt_length': 3326.952, 'perplexity': 1.04, 'retrieval_consistency': 28.988, 'syntax_error_percent': 0.202, 'semantic_error_percent': 0.381},
            'retrieved_top_pad_diff_4000': {'pass@1': 0.31, 'prompt_length': 4001.19, 'perplexity': 1.049, 'retrieval_consistency': 23.107, 'syntax_error_percent': 0.226, 'semantic_error_percent': 0.393},
            'retrieved_top_pad_repeat_diff_4000': {'pass@1': 0.262, 'prompt_length': 4237.381, 'perplexity': 1.048, 'retrieval_consistency': 24.512, 'syntax_error_percent': 0.238, 'semantic_error_percent': 0.381},
            'retrieved_top_pad_dummy_4000': {'pass@1': 0.298, 'prompt_length': 4050.774, 'perplexity': 1.05, 'retrieval_consistency': 26.476, 'syntax_error_percent': 0.226, 'semantic_error_percent': 0.417},
            'retrieved_top_pad_ellipsis_4000': {'pass@1': 0.298, 'prompt_length': 4046.94, 'perplexity': 1.055, 'syntax_error_percent': 0.25, 'semantic_error_percent': 0.417}
        },
        'irrelevant': {
            'random': {'pass@1': 0.262, 'prompt_length': 402.667, 'perplexity': 1.061, 'retrieval_consistency': 0.024, 'syntax_error_percent': 0.226, 'semantic_error_percent': 0.381},
            # 'random_500': {'pass@1': 0.321, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 491.19},
            # 'random_2000': {'pass@1': 0.321, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 1997.595},
            'random_4000': {'pass@1': 0.357, 'prompt_length': 3990.845, 'perplexity': 1.043, 'retrieval_consistency': 0.036, 'syntax_error_percent': 0.274, 'semantic_error_percent': 0.393},
            # 'random_8000': {'pass@1': 0.333, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 7983.607}
            'random_repeat_4000': {'pass@1': 0.357, 'prompt_length': 3454.333, 'perplexity': 1.044, 'retrieval_consistency': 0.0, 'syntax_error_percent': 0.25, 'semantic_error_percent': 0.393},
            'diff': {'pass@1': 0.333, 'prompt_length': 814.833, 'perplexity': 1.047, 'syntax_error_percent': 0.321, 'semantic_error_percent': 0.429},
            'diff_4000': {'pass@1': 0.357, 'prompt_length': 3966.048, 'perplexity': 1.055, 'syntax_error_percent': 0.321, 'semantic_error_percent': 0.393},
            'diff_repeat_4000': {'pass@1': 0.31, 'prompt_length': 3920.881, 'perplexity': 1.048, 'syntax_error_percent': 0.393, 'semantic_error_percent': 0.393},
            'dummy': {'pass@1': 0.31, 'prompt_length': 814.321, 'perplexity': 1.048, 'syntax_error_percent': 0.321, 'semantic_error_percent': 0.417},
            'dummy_4000': {'pass@1': 0.31, 'prompt_length': 3967.869, 'perplexity': 1.047, 'syntax_error_percent': 0.31, 'semantic_error_percent': 0.405},
            'ellipsis': None,
            'ellipsis_4000': {'pass@1': 0.274, 'prompt_length': 3957.464, 'perplexity': 1.063, 'syntax_error_percent': 0.393, 'semantic_error_percent': 0.393},
        },
        'none': {
            'none': {'pass@1': 0.226, 'prompt_length': 66.369, 'perplexity': 1.046, 'syntax_error_percent': 0.417, 'semantic_error_percent': 0.393},
            'none_pad_random_4000': {'pass@1': 0.286, 'prompt_length': 4000.476, 'perplexity': 1.045, 'syntax_error_percent': 0.345, 'semantic_error_percent': 0.357},
            'none_pad_repeat_random_4000': {'pass@1': 0.31, 'prompt_length': 4000.738, 'perplexity': 1.041, 'syntax_error_percent': 0.286, 'semantic_error_percent': 0.44},
            'none_pad_diff_4000': {'pass@1': 0.321, 'prompt_length': 4000.94, 'perplexity': 1.055, 'syntax_error_percent': 0.381, 'semantic_error_percent': 0.393},
            'none_pad_repeat_diff_4000': {'pass@1': 0.274, 'prompt_length': 4000.821, 'perplexity': 1.053, 'syntax_error_percent': 0.357, 'semantic_error_percent': 0.429},
            'none_pad_dummy_4000': {'pass@1': 0.298, 'prompt_length': 4000.929, 'perplexity': 1.055, 'syntax_error_percent': 0.393, 'semantic_error_percent': 0.393},
            'none_pad_ellipsis': {'pass@1': 0.262, 'prompt_length': 4002.94, 'perplexity': 1.051, 'syntax_error_percent': 0.345, 'semantic_error_percent': 0.357}
        },
    },
    'DS1000': {
        'oracle': {
            "oracle": {'pass@1': 0.356, 'prompt_length': 2172.236, 'perplexity': 1.033, 'retrieval_consistency': 1.287, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.191},
            'oracle_repeat_4000': {'pass@1': 0.376, 'prompt_length': 3966.675, 'perplexity': 1.034, 'retrieval_consistency': 6.745, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.185},
            'oracle_pad_random_4000': {'pass@1': 0.369, 'prompt_length': 4083.134, 'perplexity': 1.036, 'retrieval_consistency': 5.694, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.274},
            'oracle_pad_repeat_random_4000': {'pass@1': 0.358, 'prompt_length': 3951.809, 'perplexity': 1.037, 'retrieval_consistency': 6.541, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.242},
            'oracle_pad_diff_4000': {'pass@1': 0.388, 'prompt_length': 4049.287, 'perplexity': 1.038, 'retrieval_consistency': 6.439, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.236},
            'oracle_pad_repeat_diff_4000': {'pass@1': 0.392, 'prompt_length': 4321.376, 'perplexity': 1.038, 'retrieval_consistency': 5.758, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.242},
            'oracle_pad_dummy_4000': {'pass@1': 0.387, 'prompt_length': 4051.497, 'perplexity': 1.037, 'retrieval_consistency': 6.51, 'syntax_error_percent': 0.013, 'semantic_error_percent': 0.204},
            'oracle_pad_ellipsis_4000': {'pass@1': 0.373, 'prompt_length': 4036.873, 'perplexity': 1.042, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.21}
        },
        'distracting': {
            "distracting": {'pass@1': 0.264, 'prompt_length': 1898.764, 'perplexity': 1.037, 'retrieval_consistency': 0.573, 'syntax_error_percent': 0.019, 'semantic_error_percent': 0.363},
            'distracting_repeat_4000': {'pass@1': 0.319, 'prompt_length': 3953.083, 'perplexity': 1.036, 'retrieval_consistency': 3.917, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.331},
            'distracting_pad_random_4000': {'pass@1': 0.327, 'prompt_length': 4103.822, 'perplexity': 1.043, 'retrieval_consistency': 1.217, 'syntax_error_percent': 0.013, 'semantic_error_percent': 0.369},
            'distracting_pad_repeat_random_4000': {'pass@1': 0.316, 'prompt_length': 3987.376, 'perplexity': 1.04, 'retrieval_consistency': 1.675, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.344},
            'distracting_pad_diff_4000': {'pass@1': 0.321, 'prompt_length': 4086.783, 'perplexity': 1.043, 'retrieval_consistency': 1.497, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.35},
            'distracting_pad_repeat_diff_4000': {'pass@1': 0.322, 'prompt_length': 4266.707, 'perplexity': 1.044, 'retrieval_consistency': 1.58, 'syntax_error_percent': 0.013, 'semantic_error_percent': 0.35},
            'distracting_pad_dummy_4000': {'pass@1': 0.349, 'prompt_length': 4091.707, 'perplexity': 1.037, 'retrieval_consistency': 3.815, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.325},
            'distracting_pad_ellipsis_4000': {'pass@1': 0.292, 'prompt_length': 4076.783, 'perplexity': 1.047, 'syntax_error_percent': 0.025, 'semantic_error_percent': 0.369}
        },
        'retrieved_top': {
            'retrieved_top': {'pass@1': 0.339, 'prompt_length': 3139.516, 'perplexity': 1.039, 'retrieval_consistency': 0.911, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.369},
            'retrieved_top_repeat_4000': {'pass@1': 0.295, 'prompt_length': 4015.102, 'perplexity': 1.036, 'retrieval_consistency': 2.185, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.325},
            'retrieved_top_pad_random_4000': {'pass@1': 0.321, 'prompt_length': 4270.732, 'perplexity': 1.036, 'retrieval_consistency': 1.631, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.325},
            'retrieved_top_pad_repeat_random_4000': {'pass@1': 0.363, 'prompt_length': 4246.331, 'perplexity': 1.036, 'retrieval_consistency': 1.624, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.318},
            'retrieved_top_pad_diff_4000': {'pass@1': 0.343, 'prompt_length': 4260.822, 'perplexity': 1.037, 'retrieval_consistency': 1.631, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.357},
            'retrieved_top_pad_repeat_diff_4000': {'pass@1': 0.349, 'prompt_length': 5533.497, 'perplexity': 1.043, 'retrieval_consistency': 1.662, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.344},
            'retrieved_top_pad_dummy_4000': {'pass@1': 0.317, 'prompt_length': 4262.567, 'perplexity': 1.039, 'retrieval_consistency': 2.121, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.325},
            'retrieved_top_pad_ellipsis_4000': {'pass@1': 0.312, 'prompt_length': 4255.115, 'perplexity': 1.039, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.35}
        },
        'irrelevant': {
            'random': {'pass@1': 0.351, 'prompt_length': 939.217, 'perplexity': 1.038, 'retrieval_consistency': 0.0, 'syntax_error_percent': 0.019, 'semantic_error_percent': 0.401},
            'random_4000': {'pass@1': 0.371, 'prompt_length': 3999.376, 'perplexity': 1.041, 'retrieval_consistency': 0.064, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.357},
            'random_repeat_4000': {'pass@1': 0.424, 'prompt_length': 3726.051, 'perplexity': 1.033, 'retrieval_consistency': 0.0, 'syntax_error_percent': 0.013, 'semantic_error_percent': 0.312},
            'diff': {'pass@1': 0.399, 'prompt_length': 2193.631, 'perplexity': 1.039, 'syntax_error_percent': 0.019, 'semantic_error_percent': 0.318},
            'diff_4000': {'pass@1': 0.378, 'prompt_length': 4030.567, 'perplexity': 1.043, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.318},
            'diff_repeat_4000': {'pass@1': 0.407, 'prompt_length': 3970.07, 'perplexity': 1.036, 'syntax_error_percent': 0.013, 'semantic_error_percent': 0.299},
            'dummy': {'pass@1': 0.356, 'prompt_length': 2192.28, 'perplexity': 1.037, 'syntax_error_percent': 0.019, 'semantic_error_percent': 0.344},
            'dummy_4000': {'pass@1': 0.369, 'prompt_length': 4033.312, 'perplexity': 1.036, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.312},
            'ellipsis': None,
            'ellipsis_4000': {'pass@1': 0.368, 'prompt_length': 3974.783, 'perplexity': 1.053, 'syntax_error_percent': 0.013, 'semantic_error_percent': 0.287},
        },
        'none': {
            'none': {'pass@1': 0.367, 'prompt_length': 413.535, 'perplexity': 1.044, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.312},
            'none_pad_random_4000': {'pass@1': 0.323, 'prompt_length': 4000.567, 'perplexity': 1.053, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.338},
            'none_pad_repeat_random_4000': {'pass@1': 0.285, 'prompt_length': 4000.771, 'perplexity': 1.047, 'syntax_error_percent': 0.025, 'semantic_error_percent': 0.357},
            'none_pad_diff_4000': {'pass@1': 0.357, 'prompt_length': 4000.86, 'perplexity': 1.05, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.287},
            'none_pad_repeat_diff_4000': {'pass@1': 0.364, 'prompt_length': 4000.898, 'perplexity': 1.051, 'syntax_error_percent': 0.019, 'semantic_error_percent': 0.306},
            'none_pad_dummy_4000': {'pass@1': 0.343, 'prompt_length': 4000.924, 'perplexity': 1.051, 'syntax_error_percent': 0.025, 'semantic_error_percent': 0.325},
            'none_pad_ellipsis': {'pass@1': 0.332, 'prompt_length': 4002.994, 'perplexity': 1.05, 'syntax_error_percent': 0.013, 'semantic_error_percent': 0.306},
        }
    },
    'pandas_numpy_eval': {
        'oracle': {
            "oracle": {'pass@1': 0.784, 'prompt_length': 1164.731, 'perplexity': 1.026, 'retrieval_consistency': 1.114, 'syntax_error_percent': 0.036, 'semantic_error_percent': 0.126},
            'oracle_repeat_4000': {'pass@1': 0.772, 'prompt_length': 4006.144, 'perplexity': 1.033, 'retrieval_consistency': 8.425, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.126},
            'oracle_pad_random_4000': {'pass@1': 0.802, 'prompt_length': 4181.228, 'perplexity': 1.029, 'retrieval_consistency': 7.228, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.174},
            'oracle_pad_repeat_random_4000': {'pass@1': 0.808, 'prompt_length': 3629.82, 'perplexity': 1.028, 'retrieval_consistency': 7.856, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.144},
            'oracle_pad_diff_4000': {'pass@1': 0.814, 'prompt_length': 4126.677, 'perplexity': 1.029, 'retrieval_consistency': 7.641, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.162},
            'oracle_pad_repeat_diff_4000': {'pass@1': 0.79, 'prompt_length': 3624.671, 'perplexity': 1.027, 'retrieval_consistency': 7.725, 'syntax_error_percent': 0.024, 'semantic_error_percent': 0.156},
            'oracle_pad_dummy_4000': {'pass@1': 0.814, 'prompt_length': 4128.383, 'perplexity': 1.029, 'retrieval_consistency': 7.731, 'syntax_error_percent': 0.024, 'semantic_error_percent': 0.156},
            'oracle_pad_ellipsis_4000': {'pass@1': 0.784, 'prompt_length': 4104.503, 'perplexity': 1.032, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.144}
        },
        'distracting': {
            "distracting": {'pass@1': 0.653, 'prompt_length': 619.629, 'perplexity': 1.037, 'retrieval_consistency': 0.365, 'syntax_error_percent': 0.024, 'semantic_error_percent': 0.347},
            'distracting_repeat_4000': {'pass@1': 0.689, 'prompt_length': 3569.85, 'perplexity': 1.042, 'retrieval_consistency': 5.305, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.311},
            'distracting_pad_random_4000': {'pass@1': 0.713, 'prompt_length': 3652.862, 'perplexity': 1.031, 'retrieval_consistency': 3.629, 'syntax_error_percent': 0.012, 'semantic_error_percent': 0.281},
            'distracting_pad_repeat_random_4000': {'pass@1': 0.76, 'prompt_length': 3134.988, 'perplexity': 1.039, 'retrieval_consistency': 3.341, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.299},
            'distracting_pad_diff_4000': {'pass@1': 0.719, 'prompt_length': 3687.551, 'perplexity': 1.033, 'retrieval_consistency': 3.473, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.299},
            'distracting_pad_repeat_diff_4000': {'pass@1': 0.707, 'prompt_length': 3348.347, 'perplexity': 1.036, 'retrieval_consistency': 3.731, 'syntax_error_percent': 0.024, 'semantic_error_percent': 0.305},
            'distracting_pad_dummy_4000': {'pass@1': 0.695, 'prompt_length': 3697.982, 'perplexity': 1.037, 'retrieval_consistency': 3.766, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.317},
            'distracting_pad_ellipsis_4000': {'pass@1': 0.677, 'prompt_length': 3682.701, 'perplexity': 1.038, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.323}
        },
        'retrieved_top': {
            'retrieved_top': {'pass@1': 0.719, 'prompt_length': 1805.76, 'perplexity': 1.037, 'retrieval_consistency': 0.928, 'syntax_error_percent': 0.042, 'semantic_error_percent': 0.287},
            'retrieved_top_repeat_4000': {'pass@1': 0.713, 'prompt_length': 4012.228, 'perplexity': 1.038, 'retrieval_consistency': 2.79, 'syntax_error_percent': 0.036, 'semantic_error_percent': 0.311},
            'retrieved_top_pad_random_4000': {'pass@1': 0.743, 'prompt_length': 4130.551, 'perplexity': 1.032, 'retrieval_consistency': 2.683, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.305},
            'retrieved_top_pad_repeat_random_4000': {'pass@1': 0.754, 'prompt_length': 3828.707, 'perplexity': 1.036, 'retrieval_consistency': 2.784, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.287},
            'retrieved_top_pad_diff_4000': {'pass@1': 0.778, 'prompt_length': 4119.234, 'perplexity': 1.037, 'retrieval_consistency': 2.778, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.281},
            'retrieved_top_pad_repeat_diff_4000': {'pass@1': 0.754, 'prompt_length': 4917.281, 'perplexity': 1.038, 'retrieval_consistency': 2.778, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.281},
            'retrieved_top_pad_dummy_4000': {'pass@1': 0.766, 'prompt_length': 4123.814, 'perplexity': 1.041, 'retrieval_consistency': 2.593, 'syntax_error_percent': 0.042, 'semantic_error_percent': 0.275},
            'retrieved_top_pad_ellipsis_4000': {'pass@1': 0.772, 'prompt_length': 4108.832, 'perplexity': 1.04, 'syntax_error_percent': 0.024, 'semantic_error_percent': 0.317},
        },
        'irrelevant': {
            'random': {'pass@1': 0.754, 'prompt_length': 449.988, 'perplexity': 1.032, 'retrieval_consistency': 0.006, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.275},
            'random_4000': {'pass@1': 0.737, 'prompt_length': 3991.24, 'perplexity': 1.029, 'retrieval_consistency': 0.042, 'syntax_error_percent': 0.024, 'semantic_error_percent': 0.275},
            'random_repeat_4000': {'pass@1': 0.772, 'prompt_length': 3310.641, 'perplexity': 1.035, 'retrieval_consistency': 0.0, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.269},
            'diff': {'pass@1': 0.79, 'prompt_length': 1191.898, 'perplexity': 1.026, 'syntax_error_percent': 0.024, 'semantic_error_percent': 0.293},
            'diff_4000': {'pass@1': 0.772, 'prompt_length': 4153.234, 'perplexity': 1.027, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.287},
            'diff_repeat_4000': {'pass@1': 0.772, 'prompt_length': 3995.958, 'perplexity': 1.027, 'syntax_error_percent': 0.036, 'semantic_error_percent': 0.293},
            'dummy': {'pass@1': 0.772, 'prompt_length': 1190.766, 'perplexity': 1.026, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.293},
            'dummy_4000': {'pass@1': 0.737, 'prompt_length': 4155.186, 'perplexity': 1.028, 'syntax_error_percent': 0.036, 'semantic_error_percent': 0.281},
            'ellipsis': None,
            'ellipsis_4000': {'pass@1': 0.784, 'prompt_length': 3998.138, 'perplexity': 1.029, 'syntax_error_percent': 0.024, 'semantic_error_percent': 0.281},
        },
        'none': {
            'none': {'pass@1': 0.79, 'prompt_length': 142.186, 'perplexity': 1.028, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.281},
            'none_pad_random_4000': {'pass@1': 0.76, 'prompt_length': 4000.545, 'perplexity': 1.043, 'syntax_error_percent': 0.012, 'semantic_error_percent': 0.275},
            'none_pad_repeat_random_4000': {'pass@1': 0.725, 'prompt_length': 4000.713, 'perplexity': 1.04, 'syntax_error_percent': 0.024, 'semantic_error_percent': 0.299},
            'none_pad_diff_4000': {'pass@1': 0.79, 'prompt_length': 4000.826, 'perplexity': 1.029, 'syntax_error_percent': 0.012, 'semantic_error_percent': 0.269},
            'none_pad_repeat_diff_4000': {'pass@1': 0.76, 'prompt_length': 4000.88, 'perplexity': 1.033, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.275},
            'none_pad_dummy_4000': {'pass@1': 0.76, 'prompt_length': 4000.826, 'perplexity': 1.036, 'syntax_error_percent': 0.036, 'semantic_error_percent': 0.281},
            'none_pad_ellipsis': {'pass@1': 0.764, 'prompt_length': 4003.0, 'perplexity': 1.035, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.263}
        },
    },
}

code_pl_analysis_llama_n_1 = {
    'conala': {
        'oracle': {
            "oracle": {'pass@1': 0.298, 'prompt_length': 853.226, 'perplexity': 1.119, 'retrieval_consistency': 1.0, 'syntax_error_percent': 0.143, 'semantic_error_percent': 0.19},
            'oracle_repeat_4000': {'pass@1': 0.31, 'prompt_length': 3966.488, 'perplexity': 1.112, 'retrieval_consistency': 21.869, 'syntax_error_percent': 0.167, 'semantic_error_percent': 0.262},
            'oracle_pad_random_4000': {'pass@1': 0.298, 'prompt_length': 4000.345, 'perplexity': 1.138, 'retrieval_consistency': 19.036, 'syntax_error_percent': 0.25, 'semantic_error_percent': 0.357},
            'oracle_pad_repeat_random_4000': {'pass@1': 0.286, 'prompt_length': 3700.464, 'perplexity': 1.114, 'retrieval_consistency': 21.048, 'syntax_error_percent': 0.202, 'semantic_error_percent': 0.31},
            'oracle_pad_diff_4000': {'pass@1': 0.274, 'prompt_length': 4278.643, 'perplexity': 1.133, 'retrieval_consistency': 20.488, 'syntax_error_percent': 0.238, 'semantic_error_percent': 0.333},
            'oracle_pad_repeat_diff_4000': {'pass@1': 0.333, 'prompt_length': 3814.976, 'perplexity': 1.114, 'retrieval_consistency': 24.488, 'syntax_error_percent': 0.143, 'semantic_error_percent': 0.25},
            'oracle_pad_dummy_4000': {'pass@1': 0.321, 'prompt_length': 4278.643, 'perplexity': 1.116, 'retrieval_consistency': 24.0, 'syntax_error_percent': 0.167, 'semantic_error_percent': 0.25},
            'oracle_pad_ellipsis_4000': {'pass@1': 0.214, 'prompt_length': 4278.643, 'perplexity': 1.115, 'syntax_error_percent': 0.31, 'semantic_error_percent': 0.333}
        },
        'distracting': {
            "distracting": {'pass@1': 0.167, 'prompt_length': 217.095, 'perplexity': 1.131, 'retrieval_consistency': 0.429, 'syntax_error_percent': 0.202, 'semantic_error_percent': 0.464},
            'distracting_repeat_4000': {'pass@1': 0.19, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 3334.226, 'perplexity': 1.118, 'retrieval_consistency': 15.036, 'syntax_error_percent': 0.202, 'semantic_error_percent': 0.464},
            'distracting_pad_random_4000': {'pass@1': 0.238, 'prompt_length': 3006.536, 'perplexity': 1.143, 'retrieval_consistency': 14.024, 'syntax_error_percent': 0.286, 'semantic_error_percent': 0.417},
            'distracting_pad_repeat_random_4000': {'pass@1': 0.179, 'prompt_length': 2918.357, 'perplexity': 1.132, 'retrieval_consistency': 17.548, 'syntax_error_percent': 0.262, 'semantic_error_percent': 0.464},
            'distracting_pad_diff_4000': {'pass@1': 0.202, 'prompt_length': 3628.679, 'perplexity': 1.139, 'retrieval_consistency': 16.071, 'syntax_error_percent': 0.321, 'semantic_error_percent': 0.476},
            'distracting_pad_repeat_diff_4000': {'pass@1': 0.179, 'prompt_length': 3366.607, 'perplexity': 1.135, 'retrieval_consistency': 17.536, 'syntax_error_percent': 0.238, 'semantic_error_percent': 0.464},
            'distracting_pad_dummy_4000': {'pass@1': 0.167, 'prompt_length': 3628.798, 'perplexity': 1.141, 'retrieval_consistency': 17.69, 'syntax_error_percent': 0.298, 'semantic_error_percent': 0.464},
            'distracting_pad_ellipsis_4000': {'pass@1': 0.179, 'prompt_length': 3628.798, 'perplexity': 1.144, 'syntax_error_percent': 0.298, 'semantic_error_percent': 0.429}
        },
        'retrieved_top': {
            'retrieved_top': {'pass@1': 0.226, 'prompt_length': 542.631, 'perplexity': 1.129, 'retrieval_consistency': 0.738, 'syntax_error_percent': 0.226, 'semantic_error_percent': 0.44},
            'retrieved_top_repeat_4000': {'pass@1': 0.286, 'prompt_length': 4017.905, 'perplexity': 1.113, 'retrieval_consistency': 13.881, 'syntax_error_percent': 0.226, 'semantic_error_percent': 0.417},
            'retrieved_top_pad_random_4000': {'pass@1': 0.238, 'prompt_length': 3658.06, 'perplexity': 1.142, 'retrieval_consistency': 13.5, 'syntax_error_percent': 0.286, 'semantic_error_percent': 0.405},
            'retrieved_top_pad_repeat_random_4000': {'pass@1': 0.238, 'prompt_length': 3483.631, 'perplexity': 1.132, 'retrieval_consistency': 13.833, 'syntax_error_percent': 0.226, 'semantic_error_percent': 0.452},
            'retrieved_top_pad_diff_4000': {'pass@1': 0.226, 'prompt_length': 4351.786, 'perplexity': 1.136, 'retrieval_consistency': 17.845, 'syntax_error_percent': 0.274, 'semantic_error_percent': 0.44},
            'retrieved_top_pad_repeat_diff_4000': {'pass@1': 0.238, 'prompt_length': 4270.524, 'perplexity': 1.128, 'retrieval_consistency': 17.524, 'syntax_error_percent': 0.286, 'semantic_error_percent': 0.381},
            'retrieved_top_pad_dummy_4000': {'pass@1': 0.262, 'prompt_length': 4351.798, 'perplexity': 1.135, 'retrieval_consistency': 19.417, 'syntax_error_percent': 0.226, 'semantic_error_percent': 0.405},
            'retrieved_top_pad_ellipsis_4000': {'pass@1': 0.25, 'prompt_length': 4351.798, 'perplexity': 1.144, 'syntax_error_percent': 0.179, 'semantic_error_percent': 0.417},
        },
        'irrelevant': {
            'random': {'pass@1': 0.238, 'prompt_length': 449.75, 'perplexity': 1.131, 'retrieval_consistency': 0.024, 'syntax_error_percent': 0.226, 'semantic_error_percent': 0.429},
            'random_4000': {'pass@1': 0.25, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 4020.095, 'perplexity': 1.146, 'retrieval_consistency': 0.024, 'syntax_error_percent': 0.31, 'semantic_error_percent': 0.345},
            'random_repeat_4000': {'pass@1': 0.25, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 3613.298, 'perplexity': 1.113, 'retrieval_consistency': 1.179, 'syntax_error_percent': 0.19, 'semantic_error_percent': 0.44},
            'diff': {'pass@1': 0.19, 'prompt_length': 912.119, 'perplexity': 1.123, 'syntax_error_percent': 0.286, 'semantic_error_percent': 0.417},
            'diff_4000': {'pass@1': 0.202, 'prompt_length': 4337.536, 'perplexity': 1.139, 'syntax_error_percent': 0.298, 'semantic_error_percent': 0.452},
            'diff_repeat_4000': {'pass@1': 0.202, 'prompt_length': 3948.69, 'perplexity': 1.118, 'syntax_error_percent': 0.298, 'semantic_error_percent': 0.452},
            'dummy': {'pass@1': 0.25, 'prompt_length': 912.119, 'perplexity': 1.11, 'syntax_error_percent': 0.238, 'semantic_error_percent': 0.369},
            'dummy_4000': {'pass@1': 0.226, 'prompt_length': 4337.536, 'perplexity': 1.118, 'syntax_error_percent': 0.31, 'semantic_error_percent': 0.405},
            'ellipsis': None,
            'ellipsis_4000': {'pass@1': 0.214, 'prompt_length': 4337.536, 'perplexity': 1.129, 'syntax_error_percent': 0.452, 'semantic_error_percent': 0.536},
        },
        'none': {
            'none': {'pass@1': 0.226, 'prompt_length': 92.714, 'perplexity': 1.137, 'syntax_error_percent': 0.298, 'semantic_error_percent': 0.417},
            'none_pad_random_4000': {'pass@1': 0.202, 'prompt_length': 3995.417, 'perplexity': 1.158, 'syntax_error_percent': 0.583, 'semantic_error_percent': 0.464},
            'none_pad_repeat_random_4000': {'pass@1': 0.179, 'prompt_length': 3999.357, 'perplexity': 1.186, 'syntax_error_percent': 0.452, 'semantic_error_percent': 0.488},
            'none_pad_diff_4000': {'pass@1': 0.06, 'prompt_length': 4001.357, 'perplexity': 1.216, 'syntax_error_percent': 0.762, 'semantic_error_percent': 0.548},
            'none_pad_repeat_diff_4000': {'pass@1': 0.167, 'prompt_length': 4001.238, 'perplexity': 1.183, 'syntax_error_percent': 0.452, 'semantic_error_percent': 0.429},
            'none_pad_dummy_4000': {'pass@1': 0.143, 'prompt_length': 4001.0, 'perplexity': 1.227, 'syntax_error_percent': 0.607, 'semantic_error_percent': 0.476},
            'none_pad_ellipsis': {'pass@1': 0.012, 'prompt_length': 4000.0, 'perplexity': 1.039, 'syntax_error_percent': 0.964, 'semantic_error_percent': 0.94}
        },
    },
    'DS1000': {
        'oracle': {
            "oracle": {'pass@1': 0.199, 'prompt_length': 2270.185, 'perplexity': 1.143, 'retrieval_consistency': 1.204, 'syntax_error_percent': 0.102, 'semantic_error_percent': 0.191},
            'oracle_repeat_4000': {'pass@1': 0.188, 'prompt_length': 4005.503, 'perplexity': 1.146, 'retrieval_consistency': 6.096, 'syntax_error_percent': 0.108, 'semantic_error_percent': 0.197},
            'oracle_pad_random_4000': {'pass@1': 0.153, 'prompt_length': 4099.694, 'perplexity': 1.14, 'retrieval_consistency': 4.389, 'syntax_error_percent': 0.14, 'semantic_error_percent': 0.293},
            'oracle_pad_repeat_random_4000': {'pass@1': 0.186, 'prompt_length': 4050.261, 'perplexity': 1.145, 'retrieval_consistency': 5.573, 'syntax_error_percent': 0.134, 'semantic_error_percent': 0.268},
            'oracle_pad_diff_4000': {'pass@1': 0.151, 'prompt_length': 4213.618, 'perplexity': 1.15, 'retrieval_consistency': 4.331, 'syntax_error_percent': 0.108, 'semantic_error_percent': 0.287},
            'oracle_pad_repeat_diff_4000': {'pass@1': 0.19, 'prompt_length': 4238.045, 'perplexity': 1.152, 'retrieval_consistency': 5.586, 'syntax_error_percent': 0.089, 'semantic_error_percent': 0.28},
            'oracle_pad_dummy_4000': {'pass@1': 0.183, 'prompt_length': 4213.618, 'perplexity': 1.151, 'retrieval_consistency': 5.58, 'syntax_error_percent': 0.121, 'semantic_error_percent': 0.248},
            'oracle_pad_ellipsis_4000': {'pass@1': 0.185, 'prompt_length': 4213.618, 'perplexity': 1.138, 'syntax_error_percent': 0.134, 'semantic_error_percent': 0.261}
        },
        'distracting': {
            "distracting": {'pass@1': 0.123, 'prompt_length': 2024.369, 'perplexity': 1.153, 'retrieval_consistency': 0.764, 'syntax_error_percent': 0.115, 'semantic_error_percent': 0.414},
            'distracting_repeat_4000': {'pass@1': 0.129, 'prompt_length': 4003.58, 'perplexity': 1.151, 'retrieval_consistency': 3.07, 'syntax_error_percent': 0.089, 'semantic_error_percent': 0.395},
            'distracting_pad_random_4000': {'pass@1': 0.158, 'prompt_length': 4108.459, 'perplexity': 1.144, 'retrieval_consistency': 1.669, 'syntax_error_percent': 0.14, 'semantic_error_percent': 0.452},
            'distracting_pad_repeat_random_4000': {'pass@1': 0.132, 'prompt_length': 4082.299, 'perplexity': 1.152, 'retrieval_consistency': 2.484, 'syntax_error_percent': 0.108, 'semantic_error_percent': 0.389},
            'distracting_pad_diff_4000': {'pass@1': 0.145, 'prompt_length': 4266.873, 'perplexity': 1.146, 'retrieval_consistency': 1.586, 'syntax_error_percent': 0.089, 'semantic_error_percent': 0.395},
            'distracting_pad_repeat_diff_4000': {'pass@1': 0.168, 'prompt_length': 4228.178, 'perplexity': 1.155, 'retrieval_consistency': 1.739, 'syntax_error_percent': 0.083, 'semantic_error_percent': 0.389},
            'distracting_pad_dummy_4000': {'pass@1': 0.151, 'prompt_length': 4266.885, 'perplexity': 1.156, 'retrieval_consistency': 2.49, 'syntax_error_percent': 0.134, 'semantic_error_percent': 0.401},
            'distracting_pad_ellipsis_4000': {'pass@1': 0.124, 'prompt_length': 4266.885, 'perplexity': 1.138, 'syntax_error_percent': 0.146, 'semantic_error_percent': 0.439}
        },
        'retrieved_top': {
            'retrieved_top': {'pass@1': 0.082, 'prompt_length': 3275.248, 'perplexity': 1.138, 'retrieval_consistency': 1.268, 'syntax_error_percent': 0.134, 'semantic_error_percent': 0.401},
            'retrieved_top_repeat_4000': {'pass@1': 0.112, 'prompt_length': 4037.841, 'perplexity': 1.141, 'retrieval_consistency': 2.28, 'syntax_error_percent': 0.146, 'semantic_error_percent': 0.42},
            'retrieved_top_pad_random_4000': {'pass@1': 0.125, 'prompt_length': 4278.229, 'perplexity': 1.141, 'retrieval_consistency': 1.885, 'syntax_error_percent': 0.153, 'semantic_error_percent': 0.42},
            'retrieved_top_pad_repeat_random_4000': {'pass@1': 0.126, 'prompt_length': 4305.771, 'perplexity': 1.143, 'retrieval_consistency': 2.051, 'syntax_error_percent': 0.153, 'semantic_error_percent': 0.408},
            'retrieved_top_pad_diff_4000': {'pass@1': 0.128, 'prompt_length': 4350.631, 'perplexity': 1.145, 'retrieval_consistency': 1.955, 'syntax_error_percent': 0.14, 'semantic_error_percent': 0.42},
            'retrieved_top_pad_repeat_diff_4000': {'pass@1': 0.112, 'prompt_length': 5478.911, 'perplexity': 1.136, 'retrieval_consistency': 1.732, 'syntax_error_percent': 0.07, 'semantic_error_percent': 0.408},
            'retrieved_top_pad_dummy_4000': {'pass@1': 0.132, 'prompt_length': 4350.631, 'perplexity': 1.144, 'retrieval_consistency': 2.057, 'syntax_error_percent': 0.14, 'semantic_error_percent': 0.382},
            'retrieved_top_pad_ellipsis_4000': {'pass@1': 0.142, 'prompt_length': 4350.631, 'perplexity': 1.142, 'syntax_error_percent': 0.134, 'semantic_error_percent': 0.395},
        },
        'irrelevant': {
            'random': {'pass@1': 0.116, 'prompt_length': 1067.21, 'perplexity': 1.161, 'retrieval_consistency': 0.07, 'syntax_error_percent': 0.108, 'semantic_error_percent': 0.427},
            'random_4000': {'pass@1': 0.12, 'prompt_length': 4026.752, 'perplexity': 1.139, 'retrieval_consistency': 0.592, 'syntax_error_percent': 0.166, 'semantic_error_percent': 0.42},
            'random_repeat_4000': {'pass@1': 0.158, 'prompt_length': 3861.159, 'perplexity': 1.145, 'retrieval_consistency': 0.395, 'syntax_error_percent': 0.089, 'semantic_error_percent': 0.389},
            'diff': {'pass@1': 0.158, 'prompt_length': 2409.949, 'perplexity': 1.154, 'syntax_error_percent': 0.07, 'semantic_error_percent': 0.369},
            'diff_4000': {'pass@1': 0.138, 'prompt_length': 4308.261, 'perplexity': 1.147, 'syntax_error_percent': 0.051, 'semantic_error_percent': 0.376},
            'diff_repeat_4000': {'pass@1': 0.148, 'prompt_length': 3977.223, 'perplexity': 1.155, 'syntax_error_percent': 0.064, 'semantic_error_percent': 0.401},
            'dummy': {'pass@1': 0.155, 'prompt_length': 2409.949, 'perplexity': 1.156, 'syntax_error_percent': 0.045, 'semantic_error_percent': 0.395},
            'dummy_4000': {'pass@1': 0.145, 'prompt_length': 4308.261, 'perplexity': 1.155, 'syntax_error_percent': 0.064, 'semantic_error_percent': 0.401},
            'ellipsis': None,
            'ellipsis_4000': {'pass@1': 0.169, 'prompt_length': 3977.223, 'perplexity': 1.135, 'syntax_error_percent': 0.115, 'semantic_error_percent': 0.408},
        },
        'none': {
            'none': {'pass@1': 0.166, 'prompt_length': 514.739, 'perplexity': 1.15, 'syntax_error_percent': 0.032, 'semantic_error_percent': 0.401},
            'none_pad_random_4000': {'pass@1': 0.183, 'prompt_length': 3996.631, 'perplexity': 1.129, 'syntax_error_percent': 0.076, 'semantic_error_percent': 0.401},
            'none_pad_repeat_random_4000': {'pass@1': 0.141, 'prompt_length': 3998.955, 'perplexity': 1.147, 'syntax_error_percent': 0.038, 'semantic_error_percent': 0.408},
            'none_pad_diff_4000': {'pass@1': 0.17, 'prompt_length': 4001.261, 'perplexity': 1.158, 'syntax_error_percent': 0.038, 'semantic_error_percent': 0.382},
            'none_pad_repeat_diff_4000': {'pass@1': 0.191, 'prompt_length': 4001.248, 'perplexity': 1.168, 'syntax_error_percent': 0.013, 'semantic_error_percent': 0.42},
            'none_pad_dummy_4000': {'pass@1': 0.16, 'prompt_length': 4001.0, 'perplexity': 1.156, 'syntax_error_percent': 0.032, 'semantic_error_percent': 0.401},
            'none_pad_ellipsis': {'pass@1': 0.11, 'prompt_length': 4000.0, 'perplexity': 1.095, 'syntax_error_percent': 0.42, 'semantic_error_percent': 0.446}
        },
    },
    'pandas_numpy_eval': {
        'oracle': {
            "oracle": {'pass@1': 0.599, 'prompt_length': 1223.281, 'perplexity': 1.124, 'retrieval_consistency': 0.964, 'syntax_error_percent': 0.054, 'semantic_error_percent': 0.198},
            'oracle_repeat_4000': {'pass@1': 0.641, 'prompt_length': 4040.15, 'perplexity': 1.132, 'retrieval_consistency': 7.263, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.174},
            'oracle_pad_random_4000': {'pass@1': 0.707, 'prompt_length': 4174.545, 'perplexity': 1.137, 'retrieval_consistency': 6.246, 'syntax_error_percent': 0.012, 'semantic_error_percent': 0.251},
            'oracle_pad_repeat_random_4000': {'pass@1': 0.677, 'prompt_length': 3803.03, 'perplexity': 1.132, 'retrieval_consistency': 6.85, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.198},
            'oracle_pad_diff_4000': {'pass@1': 0.629, 'prompt_length': 4353.192, 'perplexity': 1.137, 'retrieval_consistency': 6.168, 'syntax_error_percent': 0.048, 'semantic_error_percent': 0.228},
            'oracle_pad_repeat_diff_4000': {'pass@1': 0.647, 'prompt_length': 3532.695, 'perplexity': 1.14, 'retrieval_consistency': 6.844, 'syntax_error_percent': 0.06, 'semantic_error_percent': 0.192},
            'oracle_pad_dummy_4000': {'pass@1': 0.707, 'prompt_length': 4353.192, 'perplexity': 1.133, 'retrieval_consistency': 6.784, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.21},
            'oracle_pad_ellipsis_4000': {'pass@1': 0.605, 'prompt_length': 4353.192, 'perplexity': 1.131, 'syntax_error_percent': 0.09, 'semantic_error_percent': 0.204}
        },
        'distracting': {
            "distracting": {'pass@1': 0.491, 'prompt_length': 691.868, 'perplexity': 1.132, 'retrieval_consistency': 0.395, 'syntax_error_percent': 0.078, 'semantic_error_percent': 0.359},
            'distracting_repeat_4000': {'pass@1': 0.611, 'prompt_length': 3718.419, 'perplexity': 1.131, 'retrieval_consistency': 3.856, 'syntax_error_percent': 0.042, 'semantic_error_percent': 0.329},
            'distracting_pad_random_4000': {'pass@1': 0.653, 'prompt_length': 3754.174, 'perplexity': 1.132, 'retrieval_consistency': 2.928, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.317},
            'distracting_pad_repeat_random_4000': {'pass@1': 0.575, 'prompt_length': 3403.964, 'perplexity': 1.133, 'retrieval_consistency': 4.114, 'syntax_error_percent': 0.06, 'semantic_error_percent': 0.341},
            'distracting_pad_diff_4000': {'pass@1': 0.599, 'prompt_length': 4058.563, 'perplexity': 1.133, 'retrieval_consistency': 3.078, 'syntax_error_percent': 0.066, 'semantic_error_percent': 0.335},
            'distracting_pad_repeat_diff_4000': {'pass@1': 0.533, 'prompt_length': 3471.18, 'perplexity': 1.144, 'retrieval_consistency': 3.072, 'syntax_error_percent': 0.132, 'semantic_error_percent': 0.371},
            'distracting_pad_dummy_4000': {'pass@1': 0.593, 'prompt_length': 4058.575, 'perplexity': 1.135, 'retrieval_consistency': 3.096, 'syntax_error_percent': 0.066, 'semantic_error_percent': 0.323},
            'distracting_pad_ellipsis_4000': {'pass@1': 0.539, 'prompt_length': 4058.575, 'perplexity': 1.129, 'syntax_error_percent': 0.114, 'semantic_error_percent': 0.329}
        },
        'retrieved_top': {
            'retrieved_top': {'pass@1': 0.545, 'prompt_length': 1935.874, 'perplexity': 1.135, 'retrieval_consistency': 0.964, 'syntax_error_percent': 0.048, 'semantic_error_percent': 0.323},
            'retrieved_top_repeat_4000': {'pass@1': 0.587, 'prompt_length': 4034.784, 'perplexity': 1.135, 'retrieval_consistency': 2.671, 'syntax_error_percent': 0.036, 'semantic_error_percent': 0.341},
            'retrieved_top_pad_random_4000': {'pass@1': 0.647, 'prompt_length': 4117.599, 'perplexity': 1.135, 'retrieval_consistency': 2.275, 'syntax_error_percent': 0.054, 'semantic_error_percent': 0.299},
            'retrieved_top_pad_repeat_random_4000': {'pass@1': 0.629, 'prompt_length': 3972.623, 'perplexity': 1.131, 'retrieval_consistency': 2.341, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.305},
            'retrieved_top_pad_diff_4000': {'pass@1': 0.647, 'prompt_length': 4282.808, 'perplexity': 1.145, 'retrieval_consistency': 2.269, 'syntax_error_percent': 0.06, 'semantic_error_percent': 0.323},
            'retrieved_top_pad_repeat_diff_4000': {'pass@1': 0.653, 'prompt_length': 4920.431, 'perplexity': 1.141, 'retrieval_consistency': 2.305, 'syntax_error_percent': 0.06, 'semantic_error_percent': 0.293},
            'retrieved_top_pad_dummy_4000': {'pass@1': 0.641, 'prompt_length': 4282.82, 'perplexity': 1.136, 'retrieval_consistency': 2.371, 'syntax_error_percent': 0.036, 'semantic_error_percent': 0.323},
            'retrieved_top_pad_ellipsis_4000': {'pass@1': 0.647, 'prompt_length': 4282.82, 'perplexity': 1.147, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.263},
        },
        'irrelevant': {
            'random': {'pass@1': 0.617, 'prompt_length': 512.156, 'perplexity': 1.124, 'retrieval_consistency': 0.006, 'syntax_error_percent': 0.108, 'semantic_error_percent': 0.311},
            'random_4000': {'pass@1': 0.683, 'prompt_length': 4021.096, 'perplexity': 1.133, 'retrieval_consistency': 0.024, 'syntax_error_percent': 0.072, 'semantic_error_percent': 0.293},
            'random_repeat_4000': {'pass@1': 0.677, 'prompt_length': 3497.527, 'perplexity': 1.127, 'retrieval_consistency': 0.671, 'syntax_error_percent': 0.06, 'semantic_error_percent': 0.305},
            'diff': {'pass@1': 0.671, 'prompt_length': 1318.581, 'perplexity': 1.11, 'syntax_error_percent': 0.036, 'semantic_error_percent': 0.287},
            'diff_4000': {'pass@1': 0.665, 'prompt_length': 4448.491, 'perplexity': 1.118, 'syntax_error_percent': 0.066, 'semantic_error_percent': 0.287},
            'diff_repeat_4000': {'pass@1': 0.677, 'prompt_length': 4005.102, 'perplexity': 1.118, 'syntax_error_percent': 0.054, 'semantic_error_percent': 0.305},
            'dummy': {'pass@1': 0.617, 'prompt_length': 1318.581, 'perplexity': 1.134, 'syntax_error_percent': 0.078, 'semantic_error_percent': 0.323},
            'dummy_4000': {'pass@1': 0.665, 'prompt_length': 4448.491, 'perplexity': 1.13, 'syntax_error_percent': 0.054, 'semantic_error_percent': 0.323},
            'ellipsis': None,
            'ellipsis_4000': {'pass@1': 0.671, 'prompt_length': 4005.102, 'perplexity': 1.137, 'syntax_error_percent': 0.036, 'semantic_error_percent': 0.293},
        },
        'none': {
            'none': {'pass@1': 0.617, 'prompt_length': 187.036, 'perplexity': 1.098, 'syntax_error_percent': 0.09, 'semantic_error_percent': 0.347},
            'none_pad_random_4000': {'pass@1': 0.677, 'prompt_length': 3995.886, 'perplexity': 1.135, 'syntax_error_percent': 0.024, 'semantic_error_percent': 0.311},
            'none_pad_repeat_random_4000': {'pass@1': 0.707, 'prompt_length': 3999.024, 'perplexity': 1.138, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.287},
            'none_pad_diff_4000': {'pass@1': 0.653, 'prompt_length': 4001.347, 'perplexity': 1.127, 'syntax_error_percent': 0.066, 'semantic_error_percent': 0.293},
            'none_pad_repeat_diff_4000': {'pass@1': 0.647, 'prompt_length': 4001.275, 'perplexity': 1.142, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.305},
            'none_pad_dummy_4000': {'pass@1': 0.701, 'prompt_length': 4001.0, 'perplexity': 1.142, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.305},
            'none_pad_ellipsis': {'pass@1': 0.419, 'prompt_length': 4000.0, 'perplexity': 1.109, 'syntax_error_percent': 0.287, 'semantic_error_percent': 0.329}
        }
    },
}



qa_pl_analysis_gpt_n_1 = {
    'NQ': {
        'oracle': {
            'oracle': {'em': 0.519, 'f1': 0.676, 'prec': 0.657, 'recall': 0.82, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 228.052, 'perplexity': 1.027},
            'pl_500': {'em': 0.522, 'f1': 0.68, 'prec': 0.659, 'recall': 0.826, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 488.079, 'perplexity': 1.026},
            'pl_2000': {'em': 0.532, 'f1': 0.689, 'prec': 0.67, 'recall': 0.825, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 1981.419, 'perplexity': 1.026},
            'pl_4000': {'em': 0.535, 'f1': 0.692, 'prec': 0.674, 'recall': 0.828, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 3981.055, 'perplexity': 1.027}
        },
        'oracle_pad_ellipsis': {
            'oracle': {'em': 0.519, 'f1': 0.676, 'prec': 0.657, 'recall': 0.82, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 228.052, 'perplexity': 1.027},
            'pl_2000': {'em': 0.546, 'f1': 0.697, 'prec': 0.682, 'recall': 0.819, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 1972.479, 'perplexity': 1.026}
        },
        'oracle_pad_reverse_ellipsis': {
            'pl_2000': {'em': 0.533, 'f1': 0.686, 'prec': 0.667, 'recall': 0.829, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 1974.47, 'perplexity': 1.033},
        },
        'oracle_pad_dummy': {
            'oracle': {'em': 0.519, 'f1': 0.676, 'prec': 0.657, 'recall': 0.82, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 228.052, 'perplexity': 1.027},
            'pl_2000': {'em': 0.548, 'f1': 0.695, 'prec': 0.681, 'recall': 0.815, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 1984.206, 'perplexity': 1.028},
        },
        'oracle_pad_reverse_dummy': {
            'pl_2000': {'em': 0.543, 'f1': 0.693, 'prec': 0.678, 'recall': 0.82, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 1984.197, 'perplexity': 1.027},
        },
        'oracle_pad_diff': {
            'oracle': {'em': 0.519, 'f1': 0.676, 'prec': 0.657, 'recall': 0.82, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 228.052, 'perplexity': 1.027},
            'pl_2000': {'em': 0.551, 'f1': 0.7, 'prec': 0.688, 'recall': 0.809, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 1989.687, 'perplexity': 1.027}
        },
        'oracle_pad_reverse_diff': {
            'pl_2000': {'em': 0.53, 'f1': 0.681, 'prec': 0.665, 'recall': 0.817, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 1989.949, 'perplexity': 1.028}
        },
        'distracting': {
            'distracting': {'em': 0.077, 'f1': 0.189, 'prec': 0.195, 'recall': 0.257, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 225.648, 'perplexity': 1.05},
            'pl_2000': {'em': 0.067, 'f1': 0.177, 'prec': 0.184, 'recall': 0.238, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 1982.787, 'perplexity': 1.048},
        },
        'distracting_pad_ellipsis': {
            'distracting': {'em': 0.077, 'f1': 0.189, 'prec': 0.195, 'recall': 0.257, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 225.648, 'perplexity': 1.05},
            'pl_2000': {'em': 0.069, 'f1': 0.172, 'prec': 0.178, 'recall': 0.232, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 1973.545, 'perplexity': 1.05}
        },
        'distracting_pad_reverse_ellipsis': {
            'pl_2000': {'em': 0.077, 'f1': 0.191, 'prec': 0.199, 'recall': 0.253, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 1975.53, 'perplexity': 1.058}
        },
        'distracting_pad_dummy': {
            'distracting': {'em': 0.077, 'f1': 0.189, 'prec': 0.195, 'recall': 0.257, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 225.648, 'perplexity': 1.05},
            'pl_2000': {'em': 0.065, 'f1': 0.171, 'prec': 0.182, 'recall': 0.218, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 1985.434, 'perplexity': 1.051}
        },
        'distracting_pad_reverse_dummy': {
            'pl_2000': {'em': 0.077, 'f1': 0.184, 'prec': 0.194, 'recall': 0.233, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 1985.418, 'perplexity': 1.05}
        },
        'distracting_pad_diff': {
            'distracting': {'em': 0.077, 'f1': 0.189, 'prec': 0.195, 'recall': 0.257, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 225.648, 'perplexity': 1.05},
            'pl_2000': {'em': 0.068, 'f1': 0.175, 'prec': 0.187, 'recall': 0.221, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 1991.075, 'perplexity': 1.049}
        },
        'distracting_pad_reverse_diff': {
            'pl_2000': {'em': 0.071, 'f1': 0.177, 'prec': 0.184, 'recall': 0.233, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 1991.313, 'perplexity': 1.05}
        },
        # 'retrieved': {
        #     'retrieved':  {'em': 0.251, 'f1': 0.39, 'prec': 0.38, 'recall': 0.501, 'ret_recall': 0.472, 'oracle_percent': 0.472, 'oracle_rank': 1.0, 'prompt_length': 226.675, 'perplexity': 1.039},
        #     'pl_500': {},
        #     'pl_2000': {},
        #     'pl_4000': {},
        # },
        'retrieved_top': {
            'retrieved': {'em': 0.344, 'f1': 0.489, 'prec': 0.475, 'recall': 0.623, 'ret_recall': 0.816, 'oracle_percent': 0.242, 'oracle_rank': 2.277, 'prompt_length': 1460.296, 'perplexity': 1.039},
            'pl_500': None,
            'pl_2000': {'em': 0.35, 'f1': 0.494, 'prec': 0.48, 'recall': 0.621, 'ret_recall': 0.816, 'oracle_percent': 0.268, 'oracle_rank': 2.277, 'prompt_length': 1982.521, 'perplexity': 1.037},
            'pl_4000': {'em': 0.34, 'f1': 0.489, 'prec': 0.474, 'recall': 0.619, 'ret_recall': 0.816, 'oracle_percent': 0.248, 'oracle_rank': 2.277, 'prompt_length': 3980.523, 'perplexity': 1.039}
        },  # top10 for QA, top5 for Code
        'retrieved_top_pad_ellipsis': {
            'retrieved': {'em': 0.344, 'f1': 0.489, 'prec': 0.475, 'recall': 0.623, 'ret_recall': 0.816, 'oracle_percent': 0.242, 'oracle_rank': 2.277, 'prompt_length': 1460.296, 'perplexity': 1.039},
            'pl_2000': {},
            'pl_4000': {'em': 0.345, 'f1': 0.492, 'prec': 0.48, 'recall': 0.615, 'ret_recall': 0.816, 'oracle_percent': 0.248, 'oracle_rank': 2.277, 'prompt_length': 3967.302, 'perplexity': 1.04}
        },
        'retrieved_top_pad_reverse_ellipsis': {
            'pl_4000': {'em': 0.347, 'f1': 0.491, 'prec': 0.48, 'recall': 0.614, 'ret_recall': 0.816, 'oracle_percent': 0.248, 'oracle_rank': 2.277, 'prompt_length': 3969.286, 'perplexity': 1.043}
        },
        'retrieved_top_pad_diff': {
            'retrieved': {'em': 0.344, 'f1': 0.489, 'prec': 0.475, 'recall': 0.623, 'ret_recall': 0.816, 'oracle_percent': 0.242, 'oracle_rank': 2.277, 'prompt_length': 1460.296, 'perplexity': 1.039},
            'pl_2000': {},
            'pl_4000': {'em': 0.365, 'f1': 0.504, 'prec': 0.496, 'recall': 0.608, 'ret_recall': 0.816, 'oracle_percent': 0.248, 'oracle_rank': 2.277, 'prompt_length': 3992.205, 'perplexity': 1.045},
        },
        'retrieved_top_pad_reverse_diff': {
            'pl_4000': {'em': 0.342, 'f1': 0.482, 'prec': 0.466, 'recall': 0.617, 'ret_recall': 0.816, 'oracle_percent': 0.248, 'oracle_rank': 2.277, 'prompt_length': 3992.465, 'perplexity': 1.042}
        },
        'retrieved_top_pad_dummy': {
            'retrieved': {'em': 0.344, 'f1': 0.489, 'prec': 0.475, 'recall': 0.623, 'ret_recall': 0.816, 'oracle_percent': 0.242, 'oracle_rank': 2.277, 'prompt_length': 1460.296, 'perplexity': 1.039},
            'pl_2000': {},
            'pl_4000': {'em': 0.362, 'f1': 0.501, 'prec': 0.491, 'recall': 0.608, 'ret_recall': 0.816, 'oracle_percent': 0.248, 'oracle_rank': 2.277, 'prompt_length': 3984.278, 'perplexity': 1.044},
        },
        'retrieved_top_pad_reverse_dummy': {
            'pl_2000': {'em': 0.354, 'f1': 0.497, 'prec': 0.488, 'recall': 0.611, 'ret_recall': 0.816, 'oracle_percent': 0.248, 'oracle_rank': 2.277, 'prompt_length': 3984.262, 'perplexity': 1.039}
        },
        'random': {
            'random': {'em': 0.164, 'f1': 0.265, 'prec': 0.263, 'recall': 0.355, 'ret_recall': 0.003, 'oracle_percent': 0.003, 'oracle_rank': 1.0, 'prompt_length': 230.421, 'perplexity': 1.077},
            'pl_500': {'em': 0.149, 'f1': 0.248, 'prec': 0.242, 'recall': 0.345, 'ret_recall': 0.011, 'oracle_percent': 0.004, 'oracle_rank': 1.619, 'prompt_length': 496.366},
            'pl_2000': {'em': 0.19, 'f1': 0.301, 'prec': 0.303, 'recall': 0.387, 'ret_recall': 0.035, 'oracle_percent': 0.004, 'oracle_rank': 6.507, 'prompt_length': 1976.963, 'perplexity': 1.083},
            'pl_4000': {'em': 0.182, 'f1': 0.292, 'prec': 0.293, 'recall': 0.379, 'ret_recall': 0.059, 'oracle_percent': 0.003, 'oracle_rank': 12.205, 'prompt_length': 3976.886}
        },
        'random_repeat': {
            'random': {'em': 0.164, 'f1': 0.265, 'prec': 0.263, 'recall': 0.355, 'ret_recall': 0.003, 'oracle_percent': 0.003, 'oracle_rank': 1.0, 'prompt_length': 230.421, 'perplexity': 1.077},
            'pl_2000': {'em': 0.167, 'f1': 0.269, 'prec': 0.266, 'recall': 0.358, 'ret_recall': 0.003, 'oracle_percent': 0.003, 'oracle_rank': 1.0, 'prompt_length': 1979.194, 'perplexity': 1.076}
        },
        'irrelevant_diff': {
            'irrelevant_diff': {'em': 0.142, 'f1': 0.2265, 'prec': 0.224, 'recall': 0.307, 'prompt_length': 228.579, 'perplexity': 1.088},
            'pl_500': {'em': 0.125, 'f1': 0.21, 'prec': 0.204, 'recall': 0.303, 'prompt_length': 489.673},
            'pl_2000': {'em': 0.161, 'f1': 0.265, 'prec': 0.261, 'recall': 0.362, 'prompt_length': 1987.573, 'perplexity': 1.094},
            'pl_4000': {'em': 0.156, 'f1': 0.258, 'prec': 0.257, 'recall': 0.35, 'prompt_length': 3999.585}
        },
        'irrelevant_diff_repeat': {
            'irrelevant_diff': {'em': 0.142, 'f1': 0.2265, 'prec': 0.224, 'recall': 0.307, 'prompt_length': 228.579, 'perplexity': 1.088},
            'pl_2000': {'em': 0.192, 'f1': 0.299, 'prec': 0.299, 'recall': 0.391, 'prompt_length': 1981.436, 'perplexity': 1.079}
        },
        'irrelevant_dummy': {
            'irrelevant_dummy': {'em': 0.12, 'f1': 0.198, 'prec': 0.1875, 'recall': 0.288, 'prompt_length': 228.262, 'perplexity': 1.087},
            'pl_500': {'em': 0.111, 'f1': 0.197, 'prec': 0.186, 'recall': 0.296, 'prompt_length': 488.71},
            'pl_2000': {'em': 0.154, 'f1': 0.254, 'prec': 0.246, 'recall': 0.35, 'prompt_length': 1984.417, 'perplexity': 1.087},
            'pl_4000': {'em': 0.159, 'f1': 0.265, 'prec': 0.258, 'recall': 0.367, 'prompt_length': 3987.259}
        },
        'ellipsis': {
            'ellipsis': {'em': 0.257, 'f1': 0.386, 'prec': 0.382, 'recall': 0.505, 'prompt_length': 229.344, 'perplexity': 1.071},
            'pl_500': {'em': 0.239, 'f1': 0.371, 'prec': 0.366, 'recall': 0.489, 'prompt_length': 490.941, 'perplexity': 1.073},
            'pl_2000': {'em': 0.227, 'f1': 0.353, 'prec': 0.351, 'recall': 0.459, 'prompt_length': 1973.771, 'perplexity': 1.09}
        },   # pad potential doc with ellipsis
        # 'ellipsis_and_pretend': {
        #
        # },   # pad doc with ellipsis and pretend this to be a document that contains information
        # 'self_pretend': {
        #
        # },   # ask llm to pretend that there are oracle document
        'none_pad_ellipsis': {
            'none': {'em': 0.247, 'f1': 0.403, 'prec': 0.381, 'recall': 0.603, 'prompt_length': 64.483, 'perplexity': 1.064},
            'pl_500': {},
            'pl_2000': {'em': 0.235, 'f1': 0.392, 'prec': 0.37, 'recall': 0.583, 'prompt_length': 2003.0, 'perplexity': 1.062},
            'pl_4000': {}
        },
        'none_pad_dummy': {
            'none': {'em': 0.247, 'f1': 0.403, 'prec': 0.381, 'recall': 0.603, 'prompt_length': 64.483, 'perplexity': 1.064},
            'pl_2000': {'em': 0.231, 'f1': 0.381, 'prec': 0.359, 'recall': 0.568, 'prompt_length': 2000.898, 'perplexity': 1.069},
        },
        'none_pad_diff': {
            'none': {'em': 0.247, 'f1': 0.403, 'prec': 0.381, 'recall': 0.603, 'prompt_length': 64.483, 'perplexity': 1.064},
            'pl_2000': {'em': 0.211, 'f1': 0.34, 'prec': 0.331, 'recall': 0.484, 'prompt_length': 2000.56, 'perplexity': 1.094}
        },
        'self_generate': {'em': 0.007, 'f1': 0.177, 'prec': 0.11, 'recall': 0.648, 'prompt_length': 88.483, 'perplexity': 1.129},   # let llm generate documents, and then answer the question
    },
    'TriviaQA': {
        'oracle': {},
        'random': {},
        'irrelevant_diff': {},
        'irrelevant_dummy': {}
    },
    'hotpotQA': {
        'oracle': {},
        'random': {},
        'irrelevant_diff': {},
        'irrelevant_dummy': {}
    }
}

qa_pl_analysis_llama_n_1 = {
    'NQ': {
        'oracle': {
            'oracle': None,
            'pl_500': None,
            'pl_1000': None,
            'pl_2000': None
        },
        'random': {
            'oracle': None,
            'pl_500': None,
            'pl_1000': None,
            'pl_2000': None
        },
        'irrelevant_diff': {
            'oracle': None,
            'pl_500': None,
            'pl_1000': None,
            'pl_2000': None
        },
        'irrelevant_dummy': {
            'oracle': None,
            'pl_500': None,
            'pl_1000': None,
            'pl_2000': None
        }
    },
    'TriviaQA': {
        'oracle': {},
        'random': {},
        'irrelevant_diff': {},
        'irrelevant_dummy': {}
    },
    'hotpotQA': {
        'oracle': {},
        'random': {},
        'irrelevant_diff': {},
        'irrelevant_dummy': {}
    }
}


prompt_method_gpt = {
    'NQ': {
        '0shot': {'em': 0.344, 'f1': 0.491, 'prec': 0.476, 'recall': 0.622, 'has_answer': 0.543, 'prompt_length': 1460.296, 'perplexity': 1.039},
        '3shot': {'em': 0.29, 'f1': 0.426, 'prec': 0.41, 'recall': 0.57, 'has_answer': 0.492, 'prompt_length': 1919.292, 'perplexity': 1.088},   # refuse to answer: 283
        'RaR': {'em': 0.0, 'f1': 0.025, 'prec': 0.013, 'recall': 0.784, 'has_answer': 0.692, 'prompt_length': 1421.296, 'perplexity': 1.177},
        'cot': {'em': 0.403, 'f1': 0.531, 'prec': 0.536, 'recall': 0.593, 'has_answer': 0.51, 'prompt_length': 2066.291, 'perplexity': 1.142},
        'self-consistency': None,
        'least_to_most': {'em': 0.367, 'f1': 0.518, 'prec': 0.51, 'recall': 0.614, 'has_answer': 0.528, 'prompt_length': 2692.291, 'perplexity': 1.105},
        'plan_and_solve': {'em': 0.188, 'f1': 0.315, 'prec': 0.289, 'recall': 0.578, 'has_answer': 0.495, 'prompt_length': 1443.296, 'perplexity': 1.154},
        'self-refine': None,
        'con': None,
        'ir-cot': None,
        'flare': None,
    },
    'TriviaQA': {
        '0shot': {'em': 0.67, 'f1': 0.752, 'prec': 0.729, 'recall': 0.837, 'has_answer': 0.818, 'prompt_length': 1500.321, 'perplexity': 1.023},
        '3shot': {'em': 0.499, 'f1': 0.595, 'prec': 0.56, 'recall': 0.799, 'has_answer': 0.775, 'prompt_length': 1959.313, 'perplexity': 1.088},    # refuse to answer: 181
        'RaR': {'em': 0.0, 'f1': 0.033, 'prec': 0.017, 'recall': 0.922, 'has_answer': 0.89, 'prompt_length': 1461.321, 'perplexity': 1.178},
        'cot': {'em': 0.682, 'f1': 0.753, 'prec': 0.737, 'recall': 0.824, 'has_answer': 0.797, 'prompt_length': 2106.314, 'perplexity': 1.156},
        'self-consistency': None,
        'least_to_most': {'em': 0.663, 'f1': 0.753, 'prec': 0.726, 'recall': 0.845, 'has_answer': 0.818, 'prompt_length': 2732.314, 'perplexity': 1.094},
        'plan_and_solve': {'em': 0.431, 'f1': 0.524, 'prec': 0.491, 'recall': 0.785, 'has_answer': 0.749, 'prompt_length': 1483.321, 'perplexity': 1.15},
        'self-refine': None,
        'con': None,
        'ir-cot': None,
        'flare': None,
    },
    'hotpotQA': {
        '0shot': {'em': 0.295, 'f1': 0.421, 'prec': 0.426, 'recall': 0.475, 'has_answer': 0.423, 'prompt_length': 963.867, 'perplexity': 1.039},
        '3shot': {'em': 0.291, 'f1': 0.401, 'prec': 0.407, 'recall': 0.482, 'has_answer': 0.406, 'prompt_length': 1805.863, 'perplexity': 1.096},    # refuse to answer: 189
        'RaR': None,
        'cot': {'em': 0.334, 'f1': 0.447, 'prec': 0.464, 'recall': 0.533, 'has_answer': 0.434, 'prompt_length': 2007.836, 'perplexity': 1.148},
        'self-consistency': None,
        'least_to_most': {'em': 0.353, 'f1': 0.465, 'prec': 0.481, 'recall': 0.54, 'has_answer': 0.442, 'prompt_length': 2574.836, 'perplexity': 1.065},
        'plan_and_solve': {'em': 0.182, 'f1': 0.316, 'prec': 0.293, 'recall': 0.525, 'has_answer': 0.461, 'prompt_length': 946.841, 'perplexity': 1.123},
        'self-refine': None,
        'con': None,
        'ir-cot': None,
        'flare': None,
    },
    'conala': {
        '0shot': {'pass@1': 0.274, 'prompt_length': 484.774, 'perplexity': 1.046, 'retrieval_consistency': 0.81, 'syntax_error_percent': 0.202, 'semantic_error_percent': 0.405},
        '3shot': {'pass@1': 0.345, 'prompt_length': 2578.786, 'perplexity': 1.04, 'retrieval_consistency': 0.714, 'syntax_error_percent': 0.143, 'semantic_error_percent': 0.369},
        'RaR': None,
        'cot': {'pass@1': 0.155, 'prompt_length': 2683.786, 'perplexity': 1.101, 'retrieval_consistency': 0.786, 'syntax_error_percent': 0.512, 'semantic_error_percent': 0.357},
        'self-consistency': None,
        'least_to_most': {'pass@1': 0.274, 'prompt_length': 3008.786, 'perplexity': 1.104, 'retrieval_consistency': 0.762, 'syntax_error_percent': 0.214, 'semantic_error_percent': 0.405},
        'plan_and_solve': None,
        'self-refine': None,
        'con': None,
        'ir-cot': None,
        'flare': None,
    },
    'DS1000': {
        '0shot': {'pass@1': 0.339, 'prompt_length': 3139.516, 'perplexity': 1.039, 'retrieval_consistency': 0.911, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.369},
        '3shot': {'pass@1': 0.271, 'prompt_length': 7478.924, 'perplexity': 1.084, 'retrieval_consistency': 0.955, 'syntax_error_percent': 0.051, 'semantic_error_percent': 0.299},
        'RaR': None,
        'cot': {'pass@1': 0.356, 'prompt_length': 7550.662, 'perplexity': 1.093, 'retrieval_consistency': 1.019, 'syntax_error_percent': 0.057, 'semantic_error_percent': 0.255},
        'self-consistency': None,
        'least_to_most': {'pass@1': 0.292, 'prompt_length': 7679.287, 'perplexity': 1.097, 'retrieval_consistency': 0.917, 'syntax_error_percent': 0.083, 'semantic_error_percent': 0.287},
        'plan_and_solve': None,
        'self-refine': None,
        'con': None,
        'ir-cot': None,
        'flare': None,
    },
    'pandas_numpy_eval': {
        '0shot': {'pass@1': 0.719, 'prompt_length': 1805.76, 'perplexity': 1.037, 'retrieval_consistency': 0.928, 'syntax_error_percent': 0.042, 'semantic_error_percent': 0.287},
        '3shot': {'pass@1': 0.754, 'prompt_length': 5755.431, 'perplexity': 1.053, 'retrieval_consistency': 0.874, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.287},
        'RaR': None,
        'cot': {'pass@1': 0.784, 'prompt_length': 5921.497, 'perplexity': 1.089, 'retrieval_consistency': 0.886, 'syntax_error_percent': 0.048, 'semantic_error_percent': 0.281},
        'self-consistency': None,
        'least_to_most': {'pass@1': 0.766, 'prompt_length': 6192.856, 'perplexity': 1.094, 'retrieval_consistency': 0.88, 'syntax_error_percent': 0.024, 'semantic_error_percent': 0.263},
        'plan_and_solve': {'pass@1': 0.689, 'prompt_length': 1752.802, 'perplexity': 1.11, 'retrieval_consistency': 0.934, 'syntax_error_percent': 0.072, 'semantic_error_percent': 0.323},
        'self-refine': None,
        'con': None,
        'ir-cot': None,
        'flare': None,
    },
}


prompt_method_llama = {
    'NQ': {
        '0shot': {'em': 0.064, 'f1': 0.211, 'prec': 0.16, 'recall': 0.657, 'has_answer': 0.559, 'prompt_length': 1720.924, 'perplexity': 1.076},
        '3shot': {'em': 0.353, 'f1': 0.491, 'prec': 0.497, 'recall': 0.538, 'has_answer': 0.448, 'prompt_length': 2241.923, 'perplexity': 1.112},
        'RaR': {'em': 0.0, 'f1': 0.039, 'prec': 0.021, 'recall': 0.694, 'has_answer': 0.6, 'prompt_length': 1680.924, 'perplexity': 1.093},
        'cot': {'em': 0.356, 'f1': 0.493, 'prec': 0.5, 'recall': 0.537, 'has_answer': 0.456, 'prompt_length': 2397.923, 'perplexity': 1.105},
        'self-consistency': None,
        'least_to_most': None,
        'plan_and_solve': None,
        'self-refine': None,
        'con': None,
        'ir-cot': None,
        'flare': None,
    },
    'TriviaQA': {
        '0shot': {'em': 0.202, 'f1': 0.378, 'prec': 0.309, 'recall': 0.861, 'has_answer': 0.825, 'prompt_length': 1759.983, 'perplexity': 1.068},
        '3shot': {'em': 0.675, 'f1': 0.737, 'prec': 0.722, 'recall': 0.787, 'has_answer': 0.763, 'prompt_length': 2280.983, 'perplexity': 1.13},
        'RaR': {'em': 0.0, 'f1': 0.076, 'prec': 0.045, 'recall': 0.819, 'has_answer': 0.821, 'prompt_length': 1719.983, 'perplexity': 1.094},
        'cot': {'em': 0.647, 'f1': 0.717, 'prec': 0.7, 'recall': 0.785, 'has_answer': 0.759, 'prompt_length': 2436.983, 'perplexity': 1.124},
        'self-consistency': None,
        'least_to_most': None,
        'plan_and_solve': None,
        'self-refine': None,
        'con': None,
        'ir-cot': None,
        'flare': None,
    },
    'hotpotQA': {
        '0shot': {'em': 0.046, 'f1': 0.169, 'prec': 0.125, 'recall': 0.536, 'has_answer': 0.443, 'prompt_length': 1150.41, 'perplexity': 1.06},
        '3shot': {'em': 0.308, 'f1': 0.415, 'prec': 0.432, 'recall': 0.433, 'has_answer': 0.35, 'prompt_length': 2190.41, 'perplexity': 1.088},
        'RaR': {'em': 0.0, 'f1': 0.06, 'prec': 0.035, 'recall': 0.506, 'has_answer': 0.46, 'prompt_length': 1110.41, 'perplexity': 1.082},
        'cot': {'em': 0.324, 'f1': 0.438, 'prec': 0.456, 'recall': 0.452, 'has_answer': 0.368, 'prompt_length': 2424.41, 'perplexity': 1.098},
        'self-consistency': None,
        'least_to_most': None,
        'plan_and_solve': None,
        'self-refine': None,
        'con': None,
        'ir-cot': None,
        'flare': None,
    },
    'conala': {
        '0shot': {'pass@1': 0.226, 'prompt_length': 542.631, 'perplexity': 1.129, 'retrieval_consistency': 0.738, 'syntax_error_percent': 0.226, 'semantic_error_percent': 0.44},
        '3shot': {'pass@1': 0.155, 'prompt_length': 2905.631, 'perplexity': 1.072, 'retrieval_consistency': 0.845, 'syntax_error_percent': 0.119, 'semantic_error_percent': 0.429},
        'RaR': {'pass@1': 0.119, 'prompt_length': 518.631, 'perplexity': 1.226, 'retrieval_consistency': 0.798, 'syntax_error_percent': 0.536, 'semantic_error_percent': 0.405},
        'cot': {'pass@1': 0.107, 'prompt_length': 3028.631, 'perplexity': 1.1, 'retrieval_consistency': 0.905, 'syntax_error_percent': 0.071, 'semantic_error_percent': 0.476},
        'self-consistency': None,
        'least_to_most': None,
        'plan_and_solve': None,
        'self-refine': None,
        'con': None,
        'ir-cot': None,
        'flare': None,
    },
    'DS1000': {
        '0shot': {'pass@1': 0.095, 'prompt_length': 3275.248, 'perplexity': 1.138, 'retrieval_consistency': 1.268, 'syntax_error_percent': 0.134, 'semantic_error_percent': 0.401},
        '3shot': {'pass@1': 0.219, 'prompt_length': 7672.045, 'perplexity': 1.102, 'retrieval_consistency': 0.955, 'syntax_error_percent': 0.051, 'semantic_error_percent': 0.369},
        'RaR': {'pass@1': 0.131, 'prompt_length': 3215.248, 'perplexity': 1.134, 'retrieval_consistency': 1.134, 'syntax_error_percent': 0.108, 'semantic_error_percent': 0.382},
        'cot': {'pass@1': 0.168, 'prompt_length': 7702.452, 'perplexity': 1.135, 'retrieval_consistency': 1.032, 'syntax_error_percent': 0.14, 'semantic_error_percent': 0.293},
        'self-consistency': None,
        'least_to_most': None,
        'plan_and_solve': None,
        'self-refine': None,
        'con': None,
        'ir-cot': None,
        'flare': None,
    },
    'pandas_numpy_eval': {
        '0shot': {'pass@1': 0.545, 'prompt_length': 1935.874, 'perplexity': 1.135, 'retrieval_consistency': 0.964, 'syntax_error_percent': 0.042, 'semantic_error_percent': 0.323},
        '3shot': {'pass@1': 0.623, 'prompt_length': 6285.856, 'perplexity': 1.159, 'retrieval_consistency': 0.844, 'syntax_error_percent': 0.036, 'semantic_error_percent': 0.311},
        'RaR': {'pass@1': 0.563, 'prompt_length': 1875.874, 'perplexity': 1.141, 'retrieval_consistency': 0.952, 'syntax_error_percent': 0.078, 'semantic_error_percent': 0.329},
        'cot': {'pass@1': 0.497, 'prompt_length': 6466.138, 'perplexity': 1.171, 'retrieval_consistency': 0.868, 'syntax_error_percent': 0.132, 'semantic_error_percent': 0.329},
        'self-consistency': None,
        'least_to_most': None,
        'plan_and_solve': None,
        'self-refine': None,
        'con': None,
        'ir-cot': None,
        'flare': None,
    },
}