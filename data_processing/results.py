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
        'BM25': {'em': 0.05, 'f1': 0.1786961769736965, 'prec': 0.13503604956505902, 'recall': 0.5456583333333338},
        'miniLM': {'em': 0.646, 'f1': 0.7243253494273003, 'prec': 0.7028446331778762, 'recall': 0.8075787878787877},
        'openai-embedding': {'em': 0.6705, 'f1': 0.7523535426367305, 'prec': 0.7286757299584995, 'recall': 0.8371984848484849},
        'contriever': {'em': 0.633, 'f1': 0.7104561200428963, 'prec': 0.6893711013373296, 'recall': 0.7928833333333333}
    },
    'TriviaQA': {
        'BM25': {'em': 0.1915, 'f1': 0.3660580759880785, 'prec': 0.29741838141695204, 'recall': 0.8419083333333328},
        'miniLM': {'em': 0.1745, 'f1': 0.3471259913860253, 'prec': 0.27950000384825197, 'recall': 0.8294466450216446},
        'openai-embedding': {'em': 0.202, 'f1': 0.37776666224593414, 'prec': 0.30937274681893795, 'recall': 0.8608916666666661},
        'contriever': {'em': 0.141, 'f1': 0.32077085408830275, 'prec': 0.2508410955069186, 'recall': 0.8228712301587301}
    },
    'hotpotQA': {
        'BM25': {'em': 0.0505, 'f1': 0.174383502626325, 'prec': 0.13057835958901104, 'recall': 0.532755244755245},
        'miniLM': {'em': 0.0505, 'f1': 0.15961374764570338, 'prec': 0.12033470411281898, 'recall': 0.4730590326340328},
        'openai-embedding': {'em': 0.046, 'f1': 0.1689016592149802, 'prec': 0.12540486085884342, 'recall': 0.5360708832833833},
        'contriever': {'em': 0.034, 'f1': 0.1464155973383291, 'prec': 0.10548973484569028, 'recall': 0.4808009157509159}
    },
    'conala': {
        'BM25': {'pass@1': 0.19, 'ret_recall': 0.083, 'oracle_percent': 0.019, 'oracle_rank': 2.75, 'prompt_length': 2999.786, 'perplexity': 1.132, 'retrieval_consistency': 0.548, 'syntax_error_percent': 0.226, 'semantic_error_percent': 0.488},
        'miniLM': {'pass@1': 0.214, 'ret_recall': 0.111, 'oracle_percent': 0.029, 'oracle_rank': 2.417, 'prompt_length': 732.595, 'perplexity': 1.13, 'retrieval_consistency': 1.25, 'syntax_error_percent': 0.179, 'semantic_error_percent': 0.488},
        'openai-embedding': {'pass@1': 0.226, 'ret_recall': 0.099, 'oracle_percent': 0.024, 'oracle_rank': 2.5, 'prompt_length': 542.631, 'perplexity': 1.129, 'retrieval_consistency': 0.738, 'syntax_error_percent': 0.238, 'semantic_error_percent': 0.44},
        'codeT5': {'pass@1': 0.238, 'ret_recall': 0.147, 'oracle_percent': 0.038, 'oracle_rank': 2.125, 'prompt_length': 1232.524, 'perplexity': 1.121, 'retrieval_consistency': 1.083, 'syntax_error_percent': 0.179, 'semantic_error_percent': 0.429},
    },
    'DS1000': {
        'BM25': {'pass@1': 0.072, 'ret_recall': 0.071, 'oracle_percent': 0.029, 'oracle_rank': 2.043, 'prompt_length': 4851.369, 'perplexity': 1.136, 'retrieval_consistency': 0.516, 'syntax_error_percent': 0.146, 'semantic_error_percent': 0.439},
        'miniLM': {'pass@1': 0.06, 'ret_recall': 0.109, 'oracle_percent': 0.042, 'oracle_rank': 2.121, 'prompt_length': 2582.197, 'perplexity': 1.143, 'retrieval_consistency': 1.541, 'syntax_error_percent': 0.14, 'semantic_error_percent': 0.427},
        'openai-embedding': {'pass@1': 0.069, 'ret_recall': 0.117, 'oracle_percent': 0.054, 'oracle_rank': 2.31, 'prompt_length': 3275.248, 'perplexity': 1.138, 'retrieval_consistency': 1.28, 'syntax_error_percent': 0.134, 'semantic_error_percent': 0.408},
        'codeT5': {'pass@1': 0.102, 'ret_recall': 0.024, 'oracle_percent': 0.006, 'oracle_rank': 2.6, 'prompt_length': 4172.554, 'perplexity': 1.137, 'retrieval_consistency': 0.459, 'syntax_error_percent': 0.127, 'semantic_error_percent': 0.414},
    },
    'pandas_numpy_eval': {
        'BM25': {'pass@1': 0.581, 'ret_recall': 0.131, 'oracle_percent': 0.031, 'oracle_rank': 2.846, 'prompt_length': 3984.994, 'perplexity': 1.136, 'retrieval_consistency': 0.246, 'syntax_error_percent': 0.096, 'semantic_error_percent': 0.299},
        'miniLM': {'pass@1': 0.521, 'ret_recall': 0.18, 'oracle_percent': 0.046, 'oracle_rank': 2.684, 'prompt_length': 1640.132, 'perplexity': 1.136, 'retrieval_consistency': 1.192, 'syntax_error_percent': 0.054, 'semantic_error_percent': 0.317},
        'openai-embedding': {'pass@1': 0.491, 'ret_recall': 0.222, 'oracle_percent': 0.053, 'oracle_rank': 2.068, 'prompt_length': 1935.874, 'perplexity': 1.135, 'retrieval_consistency': 0.982, 'syntax_error_percent': 0.06, 'semantic_error_percent': 0.323},
        'codeT5': {'pass@1': 0.515, 'ret_recall': 0.048, 'oracle_percent': 0.012, 'oracle_rank': 3.5, 'prompt_length': 2107.778, 'perplexity': 1.139, 'retrieval_consistency': 0.719, 'syntax_error_percent': 0.09, 'semantic_error_percent': 0.341},
    }
}


retriever_perf_gpt = {
    'NQ': {
        'BM25': {'em': 0.254, 'f1': 0.3799936204388769, 'prec': 0.37224948402243435, 'recall': 0.47907500000000014},
        'miniLM': {'em': 0.319, 'f1': 0.45539919755019664, 'prec': 0.43953505825638856, 'recall': 0.5902750000000003},
        'openai-embedding': {'em': 0.3445, 'f1': 0.49073557800297163, 'prec': 0.4763133890449864, 'recall': 0.6221250000000003},
        'contriever': {'em': 0.267, 'f1': 0.39120396494624526, 'prec': 0.38556238871240267, 'recall': 0.4977666666666666}
    },
    'TriviaQA': {
        'BM25': {'em': 0.6315, 'f1': 0.7166204887218547, 'prec': 0.6926273072795384, 'recall': 0.8031583333333334},
        'miniLM': {'em': 0.646, 'f1': 0.7243253494273003, 'prec': 0.7028446331778762, 'recall': 0.8075787878787877},
        'openai-embedding': {'em': 0.6705, 'f1': 0.7523535426367305, 'prec': 0.7286757299584995, 'recall': 0.8371984848484849},
        'contriever': {'em': 0.633, 'f1': 0.7104561200428963, 'prec': 0.6893711013373296, 'recall': 0.7928833333333333}
    },
    'hotpotQA': {
        'BM25': {'em': 0.2905, 'f1': 0.41366158394437524, 'prec': 0.4187718714133991, 'recall': 0.46423485958485977},
        'miniLM': {'em': 0.241, 'f1': 0.3584158256474385, 'prec': 0.36329800027556275, 'recall': 0.4102590298590299},
        'openai-embedding': {'em': 0.295, 'f1': 0.42051786332163515, 'prec': 0.42574595857381387, 'recall': 0.47493446275946294},
        'contriever': {'em': 0.263, 'f1': 0.380517603212683, 'prec': 0.3865593114832381, 'recall': 0.43119434731934764}
    },
    'conala': {
        'BM25': {'pass@1': 0.357, 'ret_recall': 0.083, 'oracle_percent': 0.019, 'oracle_rank': 2.75, 'prompt_length': 2958.607},
        'miniLM': {'pass@1': 0.298, 'ret_recall': 0.111, 'oracle_percent': 0.029, 'oracle_rank': 2.417, 'prompt_length': 663.631},
        'openai-embedding': {'pass@1': 0.274, 'ret_recall': 0.099, 'oracle_percent': 0.024, 'oracle_rank': 2.5, 'prompt_length': 484.774},
        'codeT5': {'pass@1': 0.345, 'ret_recall': 0.147, 'oracle_percent': 0.038, 'oracle_rank': 2.125, 'prompt_length': 1151.321}
    },
    'DS1000': {
        'BM25': {'pass@1': 0.322, 'ret_recall': 0.071, 'oracle_percent': 0.029, 'oracle_rank': 2.043, 'prompt_length': 4806.331},
        'miniLM': {'pass@1': 0.276, 'ret_recall': 0.109, 'oracle_percent': 0.042, 'oracle_rank': 2.121, 'prompt_length': 2360.783},
        'openai-embedding': {'pass@1': 0.323, 'ret_recall': 0.117, 'oracle_percent': 0.054, 'oracle_rank': 2.31, 'prompt_length': 3139.516},
        'codeT5': {'pass@1': 0.306, 'ret_recall': 0.024, 'oracle_percent': 0.006, 'oracle_rank': 2.6, 'prompt_length': 3971.35}
    },
    'pandas_numpy_eval': {
        'BM25': {'pass@1': 0.778, 'ret_recall': 0.131, 'oracle_percent': 0.031, 'oracle_rank': 2.846, 'prompt_length': 3831.647},
        'miniLM': {'pass@1': 0.725, 'ret_recall': 0.18, 'oracle_percent': 0.046, 'oracle_rank': 2.684, 'prompt_length': 1478.94},
        'openai-embedding': {'pass@1': 0.725, 'ret_recall': 0.222, 'oracle_percent': 0.053, 'oracle_rank': 2.068, 'prompt_length': 1805.76},
        'codeT5': {'pass@1': 0.772, 'ret_recall': 0.048, 'oracle_percent': 0.012, 'oracle_rank': 3.5, 'prompt_length': 1962.892}
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
        0.8: {'pass@1': 0.274, 'ret_recall': 0.794, 'oracle_percent': 0.794, 'oracle_rank': 1.602, 'prompt_length': 729.262, 'perplexity': 1.113, 'retrieval_consistency': 0.905, 'syntax_error_percent': 0.143, 'semantic_error_percent': 0.238},
        0.6: {'pass@1': 0.238, 'ret_recall': 0.599, 'oracle_percent': 0.599, 'oracle_rank': 1.586, 'prompt_length': 601.821, 'perplexity': 1.118, 'retrieval_consistency': 0.726, 'syntax_error_percent': 0.19, 'semantic_error_percent': 0.286},
        0.4: {'pass@1': 0.19, 'ret_recall': 0.389, 'oracle_percent': 0.389, 'oracle_rank': 1.614, 'prompt_length': 509.131, 'perplexity': 1.124, 'retrieval_consistency': 0.607, 'syntax_error_percent': 0.19, 'semantic_error_percent': 0.357},
        0.2: {'pass@1': 0.167, 'ret_recall': 0.198, 'oracle_percent': 0.198, 'oracle_rank': 1.633, 'prompt_length': 377.0, 'perplexity': 1.13, 'retrieval_consistency': 0.536, 'syntax_error_percent': 0.19, 'semantic_error_percent': 0.393},
        0: {'pass@1': 0.14, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 217.095, 'perplexity': 1.131, 'retrieval_consistency': 0.44, 'syntax_error_percent': 0.214, 'semantic_error_percent': 0.476}
    },
    "DS1000": {
        1: {'pass@1': 0.173, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 2.159, 'prompt_length': 2270.185, 'perplexity': 1.143, 'retrieval_consistency': 1.21, 'syntax_error_percent': 0.115, 'semantic_error_percent': 0.191},
        0.8: {'pass@1': 0.145, 'ret_recall': 0.799, 'oracle_percent': 0.799, 'oracle_rank': 2.178, 'prompt_length': 2215.07, 'perplexity': 1.145, 'retrieval_consistency': 1.191, 'syntax_error_percent': 0.089, 'semantic_error_percent': 0.236},
        0.6: {'pass@1': 0.155, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 2.139, 'prompt_length': 2123.834, 'perplexity': 1.151, 'retrieval_consistency': 1.172, 'syntax_error_percent': 0.083, 'semantic_error_percent': 0.28},
        0.4: {'pass@1': 0.152, 'ret_recall': 0.399, 'oracle_percent': 0.399, 'oracle_rank': 2.103, 'prompt_length': 2084.478, 'perplexity': 1.154, 'retrieval_consistency': 1.032, 'syntax_error_percent': 0.096, 'semantic_error_percent': 0.312},
        0.2: {'pass@1': 0.125, 'ret_recall': 0.194, 'oracle_percent': 0.194, 'oracle_rank': 2.134, 'prompt_length': 2020.943, 'perplexity': 1.154, 'retrieval_consistency': 0.904, 'syntax_error_percent': 0.134, 'semantic_error_percent': 0.338},
        0: {'pass@1': 0.098, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 2024.369, 'perplexity': 1.155, 'retrieval_consistency': 0.783, 'syntax_error_percent': 0.121, 'semantic_error_percent': 0.427}
    },
    "pandas_numpy_eval": {
        1: {'pass@1': 0.557, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.427, 'prompt_length': 1223.281, 'perplexity': 1.124, 'retrieval_consistency': 0.964, 'syntax_error_percent': 0.012, 'semantic_error_percent': 0.198},
        0.8: {'pass@1': 0.515, 'ret_recall': 0.795, 'oracle_percent': 0.795, 'oracle_rank': 1.42, 'prompt_length': 1096.006, 'perplexity': 1.125, 'retrieval_consistency': 0.796, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.222},
        0.6: {'pass@1': 0.503, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 1.348, 'prompt_length': 996.479, 'perplexity': 1.128, 'retrieval_consistency': 0.677, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.257},
        0.4: {'pass@1': 0.473, 'ret_recall': 0.395, 'oracle_percent': 0.395, 'oracle_rank': 1.326, 'prompt_length': 900.713, 'perplexity': 1.129, 'retrieval_consistency': 0.545, 'syntax_error_percent': 0.024, 'semantic_error_percent': 0.305},
        0.2: {'pass@1': 0.443, 'ret_recall': 0.199, 'oracle_percent': 0.199, 'oracle_rank': 1.2, 'prompt_length': 779.413, 'perplexity': 1.131, 'retrieval_consistency': 0.491, 'syntax_error_percent': 0.024, 'semantic_error_percent': 0.329},
        0: {'pass@1': 0.389, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 691.868, 'perplexity': 1.13, 'retrieval_consistency': 0.407, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.353}
    }
}


code_ret_recall_gpt_n_1 = {
    "conala": {
        1: {'pass@1': 0.381, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.589, 'prompt_length': 820.869, 'perplexity': 1.035, 'retrieval_consistency': 0.94, 'syntax_error_percent': 0.143, 'semantic_error_percent': 0.262},
        0.8: {'pass@1': 0.333, 'ret_recall': 0.794, 'oracle_percent': 0.794, 'oracle_rank': 1.602, 'prompt_length': 691.393, 'perplexity': 1.039, 'retrieval_consistency': 0.833, 'syntax_error_percent': 0.179, 'semantic_error_percent': 0.31},
        0.6: {'pass@1': 0.321, 'ret_recall': 0.599, 'oracle_percent': 0.599, 'oracle_rank': 1.586, 'prompt_length': 562.964, 'perplexity': 1.04, 'retrieval_consistency': 0.714, 'syntax_error_percent': 0.19, 'semantic_error_percent': 0.333},
        0.4: {'pass@1': 0.286, 'ret_recall': 0.389, 'oracle_percent': 0.389, 'oracle_rank': 1.614, 'prompt_length': 471.881, 'perplexity': 1.042, 'retrieval_consistency': 0.583, 'syntax_error_percent': 0.238, 'semantic_error_percent': 0.357},
        0.2: {'pass@1': 0.238, 'ret_recall': 0.198, 'oracle_percent': 0.198, 'oracle_rank': 1.633, 'prompt_length': 339.857, 'perplexity': 1.047, 'retrieval_consistency': 0.548, 'syntax_error_percent': 0.286, 'semantic_error_percent': 0.405},
        0: {'pass@1': 0.214, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 178.619, 'perplexity': 1.046, 'retrieval_consistency': 0.417, 'syntax_error_percent': 0.286, 'semantic_error_percent': 0.429}
    },
    'DS1000': {
        1: {'pass@1': 0.354, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 2.159, 'prompt_length': 2172.236, 'perplexity': 1.034, 'retrieval_consistency': 1.268, 'syntax_error_percent': 0.013, 'semantic_error_percent': 0.197},
        0.8: {'pass@1': 0.353, 'ret_recall': 0.799, 'oracle_percent': 0.799, 'oracle_rank': 2.178, 'prompt_length': 2111.662, 'perplexity': 1.039, 'retrieval_consistency': 1.146, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.255},
        0.6: {'pass@1': 0.326, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 2.139, 'prompt_length': 2009.433, 'perplexity': 1.037, 'retrieval_consistency': 1.038, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.287},
        0.4: {'pass@1': 0.354, 'ret_recall': 0.399, 'oracle_percent': 0.399, 'oracle_rank': 2.103, 'prompt_length': 1969.465, 'perplexity': 1.04, 'retrieval_consistency': 0.79, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.344},
        0.2: {'pass@1': 0.296, 'ret_recall': 0.194, 'oracle_percent': 0.194, 'oracle_rank': 2.134, 'prompt_length': 1900.93, 'perplexity': 1.04, 'retrieval_consistency': 0.65, 'syntax_error_percent': 0.013, 'semantic_error_percent': 0.376},
        0: {'pass@1': 0.275, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 1898.79, 'perplexity': 1.039, 'retrieval_consistency': 0.561, 'syntax_error_percent': 0.019, 'semantic_error_percent': 0.382}
    },
    'pandas_numpy_eval': {
        1: {'pass@1': 0.784, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.427, 'prompt_length': 1164.731, 'perplexity': 1.027, 'retrieval_consistency': 1.108, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.126},
        0.8: {'pass@1': 0.772, 'ret_recall': 0.795, 'oracle_percent': 0.795, 'oracle_rank': 1.42, 'prompt_length': 1032.581, 'perplexity': 1.028, 'retrieval_consistency': 0.904, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.174},
        0.6: {'pass@1': 0.754, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 1.348, 'prompt_length': 932.545, 'perplexity': 1.028, 'retrieval_consistency': 0.743, 'syntax_error_percent': 0.012, 'semantic_error_percent': 0.228},
        0.4: {'pass@1': 0.749, 'ret_recall': 0.395, 'oracle_percent': 0.395, 'oracle_rank': 1.326, 'prompt_length': 837.796, 'perplexity': 1.033, 'retrieval_consistency': 0.599, 'syntax_error_percent': 0.012, 'semantic_error_percent': 0.275},
        0.2: {'pass@1': 0.695, 'ret_recall': 0.199, 'oracle_percent': 0.199, 'oracle_rank': 1.2, 'prompt_length': 711.078, 'perplexity': 1.035, 'retrieval_consistency': 0.473, 'syntax_error_percent': 0.012, 'semantic_error_percent': 0.305},
        0: {'pass@1': 0.677, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 619.617, 'perplexity': 1.036, 'retrieval_consistency': 0.371, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.353}
    }
}

code_ret_doc_type_llama_n_1 = {
    "conala": {
        "oracle": {'pass@1': 0.298, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.589, 'prompt_length': 853.226, 'perplexity': 1.119, 'retrieval_consistency': 1.0, 'syntax_error_percent': 0.143, 'semantic_error_percent': 0.19},
        # "retrieved": {'pass@1': 0.16666666666666666},
        "distracting": {'pass@1': 0.167, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 217.095, 'perplexity': 1.131, 'retrieval_consistency': 0.429, 'syntax_error_percent': 0.214, 'semantic_error_percent': 0.464},
        "random": {'pass@1': 0.214, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 449.75, 'perplexity': 1.131, 'retrieval_consistency': 0.024, 'syntax_error_percent': 0.262, 'semantic_error_percent': 0.405},
        "irrelevant_dummy": {'pass@1': 0.226, 'prompt_length': 912.119, 'perplexity': 1.11, 'syntax_error_percent': 0.262, 'semantic_error_percent': 0.357},
        "irrelevant_diff": {'pass@1': 0.19, 'prompt_length': 912.119, 'perplexity': 1.123, 'syntax_error_percent': 0.298, 'semantic_error_percent': 0.393},
        "none": {'pass@1': 0.226, 'prompt_length': 92.714, 'perplexity': 1.137, 'syntax_error_percent': 0.31, 'semantic_error_percent': 0.417},
    },
    "DS1000": {
        "oracle": {'pass@1': 0.173, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 2.159, 'prompt_length': 2270.185, 'perplexity': 1.143, 'retrieval_consistency': 1.21, 'syntax_error_percent': 0.115, 'semantic_error_percent': 0.191},
        # "retrieved": {'pass@1': 0.08371569950517317},
        "distracting": {'pass@1': 0.097, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 2024.369, 'perplexity': 1.153, 'retrieval_consistency': 0.758, 'syntax_error_percent': 0.115, 'semantic_error_percent': 0.42},
        "random": {'pass@1': 0.093, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 1067.21, 'perplexity': 1.161, 'retrieval_consistency': 0.07, 'syntax_error_percent': 0.115, 'semantic_error_percent': 0.427},
        "irrelevant_dummy": {'pass@1': 0.119, 'prompt_length': 2409.949, 'perplexity': 1.156, 'syntax_error_percent': 0.038, 'semantic_error_percent': 0.446},
        "irrelevant_diff": {'pass@1': 0.138, 'prompt_length': 2409.949, 'perplexity': 1.154, 'syntax_error_percent': 0.07, 'semantic_error_percent': 0.395},
        "none": {'pass@1': 0.166, 'prompt_length': 514.739, 'perplexity': 1.15, 'syntax_error_percent': 0.038, 'semantic_error_percent': 0.42},
    },
    "pandas_numpy_eval": {
        "oracle": {'pass@1': 0.557, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.427, 'prompt_length': 1223.281, 'perplexity': 1.124, 'retrieval_consistency': 0.964, 'syntax_error_percent': 0.012, 'semantic_error_percent': 0.198},
        # "retrieved": {'pass@1': 0.38922155688622756},
        "distracting": {'pass@1': 0.383, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 691.868, 'perplexity': 1.132, 'retrieval_consistency': 0.407, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.353},
        "random": {'pass@1': 0.515, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 512.156, 'perplexity': 1.124, 'retrieval_consistency': 0.006, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.311},
        "irrelevant_dummy": {'pass@1': 0.533, 'prompt_length': 1318.581, 'perplexity': 1.134, 'syntax_error_percent': 0.012, 'semantic_error_percent': 0.323},
        "irrelevant_diff": {'pass@1': 0.641, 'prompt_length': 1318.581, 'perplexity': 1.11, 'syntax_error_percent': 0.06, 'semantic_error_percent': 0.287},
        "none": {'pass@1': 0.563, 'prompt_length': 187.036, 'perplexity': 1.098, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.347},
    }
}

code_ret_doc_type_gpt_n_1 = {
    "conala": {
        "oracle": {'pass@1': 0.381, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.589, 'prompt_length': 820.869, 'perplexity': 1.038, 'retrieval_consistency': 0.94, 'syntax_error_percent': 0.167, 'semantic_error_percent': 0.25},
        # "retrieved": {'pass@1': 0.21428571428571427},
        "distracting": {'pass@1': 0.226, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 178.619, 'perplexity': 1.046, 'retrieval_consistency': 0.452, 'syntax_error_percent': 0.286, 'semantic_error_percent': 0.44},
        "random": {'pass@1': 0.262, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 402.667, 'perplexity': 1.061, 'retrieval_consistency': 0.024, 'syntax_error_percent': 0.262, 'semantic_error_percent': 0.381},
        "irrelevant_dummy": {'pass@1': 0.286, 'prompt_length': 814.321, 'perplexity': 1.048, 'syntax_error_percent': 0.381, 'semantic_error_percent': 0.417},
        "irrelevant_diff": {'pass@1': 0.321, 'prompt_length': 814.833, 'perplexity': 1.047, 'syntax_error_percent': 0.369, 'semantic_error_percent': 0.405},
        "none": {'pass@1': 0.214, 'prompt_length': 66.369, 'perplexity': 1.046, 'syntax_error_percent': 0.452, 'semantic_error_percent': 0.381}
    },
    "DS1000": {
        "oracle": {'pass@1': 0.34, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 2.159, 'prompt_length': 2172.236, 'perplexity': 1.033, 'retrieval_consistency': 1.287, 'syntax_error_percent': 0.019, 'semantic_error_percent': 0.191},
        # "retrieved": {'pass@1': 0.25234407087954},
        "distracting": {'pass@1': 0.26, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 1898.764, 'perplexity': 1.037, 'retrieval_consistency': 0.58, 'syntax_error_percent': 0.025, 'semantic_error_percent': 0.357},
        "random": {'pass@1': 0.33, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 939.217, 'perplexity': 1.038, 'retrieval_consistency': 0.0, 'syntax_error_percent': 0.019, 'semantic_error_percent': 0.395},
        "irrelevant_dummy": {'pass@1': 0.342, 'prompt_length': 2192.28, 'perplexity': 1.037, 'syntax_error_percent': 0.025, 'semantic_error_percent': 0.344},
        "irrelevant_diff": {'pass@1': 0.383, 'prompt_length': 2193.631, 'perplexity': 1.039, 'syntax_error_percent': 0.013, 'semantic_error_percent': 0.312},
        "none": {'pass@1': 0.326, 'prompt_length': 413.535, 'perplexity': 1.044, 'syntax_error_percent': 0.0, 'semantic_error_percent': 0.312},
    },
    "pandas_numpy_eval": {
        "oracle": {'pass@1': 0.79, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.427, 'prompt_length': 1164.731, 'perplexity': 1.026, 'retrieval_consistency': 1.114, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.126},
        # "retrieved": {'pass@1': 0.6586826347305389},
        "distracting": {'pass@1': 0.659, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 619.629, 'perplexity': 1.037, 'retrieval_consistency': 0.365, 'syntax_error_percent': 0.012, 'semantic_error_percent': 0.347},
        "random": {'pass@1': 0.754, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 449.988, 'perplexity': 1.032, 'retrieval_consistency': 0.006, 'syntax_error_percent': 0.012, 'semantic_error_percent': 0.275},
        "irrelevant_dummy": {'pass@1': 0.766, 'prompt_length': 1190.766, 'perplexity': 1.026, 'syntax_error_percent': 0.036, 'semantic_error_percent': 0.287},
        "irrelevant_diff": {'pass@1': 0.784, 'prompt_length': 1191.898, 'perplexity': 1.026, 'syntax_error_percent': 0.03, 'semantic_error_percent': 0.287},
        "none": {'pass@1': 0.778, 'prompt_length': 142.186, 'perplexity': 1.028, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.281}
    }
}


qa_ret_recall_llama_n_1 = {
    'NQ': {
        1: {'em': 0.203, 'f1': 0.398, 'prec': 0.345, 'recall': 0.806, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 283.926, 'perplexity': 1.061},
        0.8: {'em': 0.171, 'f1': 0.347, 'prec': 0.298, 'recall': 0.716, 'ret_recall': 0.8, 'oracle_percent': 0.8, 'oracle_rank': 1.0, 'prompt_length': 283.483, 'perplexity': 1.064},
        0.6: {'em': 0.133, 'f1': 0.29, 'prec': 0.246, 'recall': 0.617, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 1.0, 'prompt_length': 282.443, 'perplexity': 1.066},
        0.4: {'em': 0.097, 'f1': 0.235, 'prec': 0.197, 'recall': 0.521, 'ret_recall': 0.4, 'oracle_percent': 0.4, 'oracle_rank': 1.0, 'prompt_length': 281.589, 'perplexity': 1.068},
        0.2: {'em': 0.067, 'f1': 0.185, 'prec': 0.153, 'recall': 0.427, 'ret_recall': 0.2, 'oracle_percent': 0.2, 'oracle_rank': 1.0, 'prompt_length': 281.065, 'perplexity': 1.071},
        0: {'em': 0.028, 'f1': 0.125, 'prec': 0.1, 'recall': 0.323, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 280.384, 'perplexity': 1.074},
    },
    "TriviaQA": {
        1.0: {'em': 0.358, 'f1': 0.523, 'prec': 0.461, 'recall': 0.904, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 295.269, 'perplexity': 1.062},
        0.8: {'em': 0.314, 'f1': 0.47, 'prec': 0.412, 'recall': 0.836, 'ret_recall': 0.8, 'oracle_percent': 0.8, 'oracle_rank': 1.0, 'prompt_length': 294.865, 'perplexity': 1.063},
        0.6: {'em': 0.263, 'f1': 0.416, 'prec': 0.361, 'recall': 0.771, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 1.0, 'prompt_length': 294.474, 'perplexity': 1.065},
        0.4: {'em': 0.217, 'f1': 0.37, 'prec': 0.317, 'recall': 0.723, 'ret_recall': 0.4, 'oracle_percent': 0.4, 'oracle_rank': 1.0, 'prompt_length': 293.97, 'perplexity': 1.066},
        0.2: {'em': 0.166, 'f1': 0.318, 'prec': 0.265, 'recall': 0.667, 'ret_recall': 0.2, 'oracle_percent': 0.2, 'oracle_rank': 1.0, 'prompt_length': 293.853, 'perplexity': 1.068},
        0.0: {'em': 0.115, 'f1': 0.264, 'prec': 0.212, 'recall': 0.61, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 293.599, 'perplexity': 1.069},
    },
    "hotpotQA": {
        1.0: {'em': 0.234, 'f1': 0.427, 'prec': 0.384, 'recall': 0.79, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.5, 'prompt_length': 371.815, 'perplexity': 1.057},
        0.8: {'em': 0.197, 'f1': 0.37, 'prec': 0.332, 'recall': 0.695, 'ret_recall': 0.8, 'oracle_percent': 0.8, 'oracle_rank': 1.495, 'prompt_length': 365.962, 'perplexity': 1.059},
        0.6: {'em': 0.157, 'f1': 0.312, 'prec': 0.275, 'recall': 0.608, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 1.492, 'prompt_length': 358.126, 'perplexity': 1.061},
        0.4: {'em': 0.121, 'f1': 0.252, 'prec': 0.218, 'recall': 0.512, 'ret_recall': 0.4, 'oracle_percent': 0.4, 'oracle_rank': 1.486, 'prompt_length': 351.218, 'perplexity': 1.064},
        0.2: {'em': 0.084, 'f1': 0.191, 'prec': 0.163, 'recall': 0.404, 'ret_recall': 0.2, 'oracle_percent': 0.2, 'oracle_rank': 1.458, 'prompt_length': 342.573, 'perplexity': 1.068},
        0.0: {'em': 0.054, 'f1': 0.134, 'prec': 0.113, 'recall': 0.293, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 334.702, 'perplexity': 1.071},
    }
}


qa_ret_recall_gpt_n_1 = {
    'NQ': {
        1.0: {'em': 0.523, 'f1': 0.681, 'prec': 0.663, 'recall': 0.822, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 228.052, 'perplexity': 1.027},
        0.8: {'em': 0.434, 'f1': 0.585, 'prec': 0.573, 'recall': 0.709, 'ret_recall': 0.8, 'oracle_percent': 0.8, 'oracle_rank': 1.0, 'prompt_length': 227.672, 'perplexity': 1.032},
        0.6: {'em': 0.345, 'f1': 0.487, 'prec': 0.48, 'recall': 0.596, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 1.0, 'prompt_length': 227.066, 'perplexity': 1.035},
        0.4: {'em': 0.258, 'f1': 0.391, 'prec': 0.39, 'recall': 0.484, 'ret_recall': 0.4, 'oracle_percent': 0.4, 'oracle_rank': 1.0, 'prompt_length': 226.524, 'perplexity': 1.041},
        0.2: {'em': 0.165, 'f1': 0.294, 'prec': 0.3, 'recall': 0.371, 'ret_recall': 0.2, 'oracle_percent': 0.2, 'oracle_rank': 1.0, 'prompt_length': 226.031, 'perplexity': 1.046},
        0.0: {'em': 0.077, 'f1': 0.192, 'prec': 0.198, 'recall': 0.262, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 225.648, 'perplexity': 1.05}
    },
    "TriviaQA": {
        1.0: {'em': 0.729, 'f1': 0.808, 'prec': 0.783, 'recall': 0.895, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 237.544, 'perplexity': 1.018},
        0.8: {'em': 0.666, 'f1': 0.743, 'prec': 0.72, 'recall': 0.827, 'ret_recall': 0.8, 'oracle_percent': 0.8, 'oracle_rank': 1.0, 'prompt_length': 237.381, 'perplexity': 1.024},
        0.6: {'em': 0.596, 'f1': 0.677, 'prec': 0.66, 'recall': 0.757, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 1.0, 'prompt_length': 237.06, 'perplexity': 1.028},
        0.4: {'em': 0.532, 'f1': 0.614, 'prec': 0.6, 'recall': 0.691, 'ret_recall': 0.4, 'oracle_percent': 0.4, 'oracle_rank': 1.0, 'prompt_length': 236.735, 'perplexity': 1.033},
        0.2: {'em': 0.458, 'f1': 0.544, 'prec': 0.532, 'recall': 0.623, 'ret_recall': 0.2, 'oracle_percent': 0.2, 'oracle_rank': 1.0, 'prompt_length': 236.626, 'perplexity': 1.037},
        0.0: {'em': 0.399, 'f1': 0.488, 'prec': 0.477, 'recall': 0.569, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 236.456, 'perplexity': 1.04}
    },
    "hotpotQA": {
        1.0: {'em': 0.528, 'f1': 0.693, 'prec': 0.708, 'recall': 0.75, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.5, 'prompt_length': 300.92, 'perplexity': 1.021},
        0.8: {'em': 0.453, 'f1': 0.604, 'prec': 0.617, 'recall': 0.659, 'ret_recall': 0.8, 'oracle_percent': 0.8, 'oracle_rank': 1.495, 'prompt_length': 296.255, 'perplexity': 1.027},
        0.6: {'em': 0.372, 'f1': 0.514, 'prec': 0.528, 'recall': 0.562, 'ret_recall': 0.6, 'oracle_percent': 0.6, 'oracle_rank': 1.492, 'prompt_length': 289.354, 'perplexity': 1.033},
        0.4: {'em': 0.309, 'f1': 0.428, 'prec': 0.436, 'recall': 0.475, 'ret_recall': 0.4, 'oracle_percent': 0.4, 'oracle_rank': 1.486, 'prompt_length': 283.193, 'perplexity': 1.039},
        0.2: {'em': 0.234, 'f1': 0.337, 'prec': 0.345, 'recall': 0.376, 'ret_recall': 0.2, 'oracle_percent': 0.2, 'oracle_rank': 1.458, 'prompt_length': 276.349, 'perplexity': 1.045},
        0.0: {'em': 0.145, 'f1': 0.232, 'prec': 0.235, 'recall': 0.268, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 269.839, 'perplexity': 1.053},
    }
}

qa_ret_doc_type_llama_n_1 = {
    'NQ': {
        "oracle": {'em': 0.203, 'f1': 0.398, 'prec': 0.345, 'recall': 0.806, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 283.926, 'perplexity': 1.061},
        # "retrieved": {'em': 0.1095, 'f1': 0.24555840009200883, 'prec': 0.20931816465800238, 'recall': 0.5241250000000001},
        "distracting": {'em': 0.028, 'f1': 0.125, 'prec': 0.1, 'recall': 0.323, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 280.384, 'perplexity': 1.074},
        "random": {'em': 0.002, 'f1': 0.073, 'prec': 0.046, 'recall': 0.281, 'ret_recall': 0.003, 'oracle_percent': 0.003, 'oracle_rank': 1.0, 'prompt_length': 284.834, 'perplexity': 1.11},
        "irrelevant_dummy": {'em': 0.005, 'f1': 0.08, 'prec': 0.052, 'recall': 0.268, 'prompt_length': 283.926, 'perplexity': 1.105},
        "irrelevant_diff": {'em': 0.003, 'f1': 0.105, 'prec': 0.065, 'recall': 0.379, 'prompt_length': 283.923, 'perplexity': 1.12},
        "none": {'em': 0.013, 'f1': 0.121, 'prec': 0.078, 'recall': 0.509, 'prompt_length': 89.734, 'perplexity': 1.087},
    },
    "TriviaQA": {
        "oracle": {'em': 0.358, 'f1': 0.523, 'prec': 0.461, 'recall': 0.904, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 295.269, 'perplexity': 1.062},
        # "retrieved": {'em': 0.298, 'f1': 0.44355579510359733, 'prec': 0.39018332836747666, 'recall': 0.7751964285714283},
        "distracting": {'em': 0.115, 'f1': 0.264, 'prec': 0.212, 'recall': 0.61, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 293.599, 'perplexity': 1.069},
        "random": {'em': 0.048, 'f1': 0.223, 'prec': 0.157, 'recall': 0.668, 'ret_recall': 0.006, 'oracle_percent': 0.006, 'oracle_rank': 1.0, 'prompt_length': 294.918, 'perplexity': 1.087},
        "irrelevant_dummy": {'em': 0.089, 'f1': 0.271, 'prec': 0.205, 'recall': 0.69, 'prompt_length': 295.269, 'perplexity': 1.081},
        "irrelevant_diff": {'em': 0.06, 'f1': 0.239, 'prec': 0.17, 'recall': 0.717, 'prompt_length': 295.264, 'perplexity': 1.097},
        "none": {'em': 0.104, 'f1': 0.279, 'prec': 0.21, 'recall': 0.791, 'prompt_length': 99.819, 'perplexity': 1.06}
    },
    "hotpotQA": {
        "oracle": {'em': 0.235, 'f1': 0.426, 'prec': 0.385, 'recall': 0.781, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.5, 'prompt_length': 371.815, 'perplexity': 1.057},
        # "retrieved": {'em': 0.1135, 'f1': 0.23571233027433397, 'prec': 0.20502697902833195, 'recall': 0.47636953046953057},
        "distracting": {'em': 0.059, 'f1': 0.136, 'prec': 0.115, 'recall': 0.291, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 334.702, 'perplexity': 1.07},
        "random": {'em': 0.037, 'f1': 0.12, 'prec': 0.093, 'recall': 0.293, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 287.745, 'perplexity': 1.085},
        "irrelevant_dummy": {'em': 0.053, 'f1': 0.139, 'prec': 0.113, 'recall': 0.304, 'prompt_length': 371.815, 'perplexity': 1.092},
        "irrelevant_diff": {'em': 0.035, 'f1': 0.122, 'prec': 0.094, 'recall': 0.312, 'prompt_length': 371.811, 'perplexity': 1.105},
        "none": {'em': 0.022, 'f1': 0.115, 'prec': 0.083, 'recall': 0.36, 'prompt_length': 101.672, 'perplexity': 1.073},
    }
}

qa_ret_doc_type_gpt_n_1 = {
    'NQ': {
        "oracle": {'em': 0.519, 'f1': 0.676, 'prec': 0.657, 'recall': 0.82, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 228.052, 'perplexity': 1.027},
        # "retrieved": {'em': 0.2505, 'f1': 0.38966929494021535, 'prec': 0.3798526426123864, 'recall': 0.5012416666666667},
        "distracting": {'em': 0.077, 'f1': 0.189, 'prec': 0.195, 'recall': 0.257, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 225.648, 'perplexity': 1.05},
        "random": {'em': 0.164, 'f1': 0.265, 'prec': 0.263, 'recall': 0.355, 'ret_recall': 0.003, 'oracle_percent': 0.003, 'oracle_rank': 1.0, 'prompt_length': 230.421, 'perplexity': 1.077},
        "irrelevant_dummy": {'em': 0.12, 'f1': 0.198, 'prec': 0.188, 'recall': 0.288, 'prompt_length': 228.262, 'perplexity': 1.087},
        "irrelevant_diff": {'em': 0.142, 'f1': 0.227, 'prec': 0.224, 'recall': 0.307, 'prompt_length': 228.579, 'perplexity': 1.088},
        "none": {'em': 0.247, 'f1': 0.403, 'prec': 0.381, 'recall': 0.603, 'prompt_length': 64.483, 'perplexity': 1.064},
    },
    "TriviaQA": {
        "oracle": {'em': 0.734, 'f1': 0.812, 'prec': 0.786, 'recall': 0.898, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 237.544, 'perplexity': 1.018},
        # "retrieved": {'em': 0.588, 'f1': 0.6785403596045625, 'prec': 0.6573662049086141, 'recall': 0.7640666666666663},
        "distracting": {'em': 0.401, 'f1': 0.489, 'prec': 0.478, 'recall': 0.569, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 236.456, 'perplexity': 1.04},
        "random": {'em': 0.65, 'f1': 0.716, 'prec': 0.697, 'recall': 0.791, 'ret_recall': 0.006, 'oracle_percent': 0.006, 'oracle_rank': 1.0, 'prompt_length': 237.699, 'perplexity': 1.04},
        "irrelevant_dummy": {'em': 0.645, 'f1': 0.714, 'prec': 0.693, 'recall': 0.807, 'prompt_length': 237.794, 'perplexity': 1.042},
        "irrelevant_diff": {'em': 0.666, 'f1': 0.728, 'prec': 0.712, 'recall': 0.797, 'prompt_length': 238.488, 'perplexity': 1.043},
        "none": {'em': 0.706, 'f1': 0.773, 'prec': 0.748, 'recall': 0.878, 'prompt_length': 71.761, 'perplexity': 1.029},
    },
    "hotpotQA": {
        "oracle": {'em': 0.521, 'f1': 0.688, 'prec': 0.702, 'recall': 0.745, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.5, 'prompt_length': 300.921, 'perplexity': 1.021},
        # "retrieved": {'em': 0.2615, 'f1': 0.38276433036531554, 'prec': 0.38988646984133074, 'recall': 0.43652234154734176},
        "distracting": {'em': 0.147, 'f1': 0.234, 'prec': 0.238, 'recall': 0.269, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 269.839, 'perplexity': 1.055},
        "random": {'em': 0.175, 'f1': 0.26, 'prec': 0.262, 'recall': 0.301, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 231.194, 'perplexity': 1.08},
        "irrelevant_dummy": {'em': 0.183, 'f1': 0.263, 'prec': 0.267, 'recall': 0.311, 'prompt_length': 303.166, 'perplexity': 1.087},
        "irrelevant_diff": {'em': 0.196, 'f1': 0.276, 'prec': 0.285, 'recall': 0.31, 'prompt_length': 304.324, 'perplexity': 1.091},
        "none": {'em': 0.202, 'f1': 0.333, 'prec': 0.336, 'recall': 0.391, 'prompt_length': 73.689, 'perplexity': 1.084},
    }
}



# todo: top1 top5 top10 top15 top20 for code llama (16k) and gpt (16k)
code_ret_doc_selection_topk_llama_n_1 = {
    'conala': {
        'top_1': {'pass@1': 0.167, 'ret_recall': 0.004, 'oracle_percent': 0.012, 'oracle_rank': 1.0, 'prompt_length': 192.631, 'perplexity': 1.133, 'retrieval_consistency': 0.321, 'syntax_error_percent': 0.19, 'semantic_error_percent': 0.5},
        'top_5': {'pass@1': 0.226, 'ret_recall': 0.099, 'oracle_percent': 0.024, 'oracle_rank': 2.5, 'prompt_length': 542.631, 'perplexity': 1.129, 'retrieval_consistency': 0.738, 'syntax_error_percent': 0.238, 'semantic_error_percent': 0.44},
        'top_10': {'pass@1': 0.214, 'ret_recall': 0.151, 'oracle_percent': 0.019, 'oracle_rank': 4.5, 'prompt_length': 968.214, 'perplexity': 1.131, 'retrieval_consistency': 1.19, 'syntax_error_percent': 0.238, 'semantic_error_percent': 0.452},
        'top_15': {'pass@1': 0.19, 'ret_recall': 0.193, 'oracle_percent': 0.018, 'oracle_rank': 6.87, 'prompt_length': 1538.417, 'perplexity': 1.132, 'retrieval_consistency': 1.524, 'syntax_error_percent': 0.298, 'semantic_error_percent': 0.44},
        'top_20': {'pass@1': 0.143, 'ret_recall': 0.211, 'oracle_percent': 0.015, 'oracle_rank': 7.68, 'prompt_length': 2045.929, 'perplexity': 1.129, 'retrieval_consistency': 1.631, 'syntax_error_percent': 0.31, 'semantic_error_percent': 0.452}
    },
    'DS1000': {
        'top_1': {'pass@1': 0.128, 'ret_recall': 0.043, 'oracle_percent': 0.108, 'oracle_rank': 1.0, 'prompt_length': 1155.414, 'perplexity': 1.155, 'retrieval_consistency': 0.401, 'syntax_error_percent': 0.096, 'semantic_error_percent': 0.414},
        'top_5': {'pass@1': 0.069, 'ret_recall': 0.117, 'oracle_percent': 0.054, 'oracle_rank': 2.31, 'prompt_length': 3275.248, 'perplexity': 1.138, 'retrieval_consistency': 1.28, 'syntax_error_percent': 0.134, 'semantic_error_percent': 0.408},
        'top_10': {'pass@1': 0.065, 'ret_recall': 0.181, 'oracle_percent': 0.041, 'oracle_rank': 4.262, 'prompt_length': 5470.917, 'perplexity': 1.132, 'retrieval_consistency': 1.758, 'syntax_error_percent': 0.146, 'semantic_error_percent': 0.439},
        'top_15': {'pass@1': 0.105, 'ret_recall': 0.21, 'oracle_percent': 0.033, 'oracle_rank': 5.692, 'prompt_length': 6466.204, 'perplexity': 1.126, 'retrieval_consistency': 2.229, 'syntax_error_percent': 0.172, 'semantic_error_percent': 0.42},
        'top_20': {'pass@1': 0.076, 'ret_recall': 0.229, 'oracle_percent': 0.027, 'oracle_rank': 6.536, 'prompt_length': 6848.389, 'perplexity': 1.124, 'retrieval_consistency': 2.777, 'syntax_error_percent': 0.217, 'semantic_error_percent': 0.42}
    },
    'pandas_numpy_eval': {
        'top_1': {'pass@1': 0.431, 'ret_recall': 0.099, 'oracle_percent': 0.12, 'oracle_rank': 1.0, 'prompt_length': 573.844, 'perplexity': 1.126, 'retrieval_consistency': 0.341, 'syntax_error_percent': 0.012, 'semantic_error_percent': 0.353},
        'top_5': {'pass@1': 0.491, 'ret_recall': 0.222, 'oracle_percent': 0.053, 'oracle_rank': 2.068, 'prompt_length': 1935.874, 'perplexity': 1.135, 'retrieval_consistency': 0.982, 'syntax_error_percent': 0.06, 'semantic_error_percent': 0.323},
        'top_10': {'pass@1': 0.587, 'ret_recall': 0.294, 'oracle_percent': 0.036, 'oracle_rank': 3.533, 'prompt_length': 3660.982, 'perplexity': 1.144, 'retrieval_consistency': 1.389, 'syntax_error_percent': 0.09, 'semantic_error_percent': 0.335},
        'top_15': {'pass@1': 0.611, 'ret_recall': 0.357, 'oracle_percent': 0.029, 'oracle_rank': 5.0, 'prompt_length': 5122.91, 'perplexity': 1.137, 'retrieval_consistency': 1.713, 'syntax_error_percent': 0.09, 'semantic_error_percent': 0.287},
        'top_20': {'pass@1': 0.581, 'ret_recall': 0.395, 'oracle_percent': 0.024, 'oracle_rank': 6.213, 'prompt_length': 6155.419, 'perplexity': 1.143, 'retrieval_consistency': 1.982, 'syntax_error_percent': 0.126, 'semantic_error_percent': 0.305}
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
        'top_1': {'pass@1': 0.179, 'ret_recall': 0.004, 'oracle_percent': 0.012, 'oracle_rank': 1.0, 'prompt_length': 157.976, 'perplexity': 1.049, 'retrieval_consistency': 0.381, 'syntax_error_percent': 0.369, 'semantic_error_percent': 0.44},
        'top_5': {'pass@1': 0.274, 'ret_recall': 0.099, 'oracle_percent': 0.024, 'oracle_rank': 2.5, 'prompt_length': 484.774, 'perplexity': 1.046, 'retrieval_consistency': 0.81, 'syntax_error_percent': 0.214, 'semantic_error_percent': 0.405},
        'top_10': {'pass@1': 0.286, 'ret_recall': 0.151, 'oracle_percent': 0.019, 'oracle_rank': 4.5, 'prompt_length': 875.19, 'perplexity': 1.043, 'retrieval_consistency': 1.333, 'syntax_error_percent': 0.25, 'semantic_error_percent': 0.381},
        'top_15': {'pass@1': 0.25, 'ret_recall': 0.193, 'oracle_percent': 0.018, 'oracle_rank': 6.87, 'prompt_length': 1405.833, 'perplexity': 1.044, 'retrieval_consistency': 1.583, 'syntax_error_percent': 0.262, 'semantic_error_percent': 0.405},
        'top_20': {'pass@1': 0.25, 'ret_recall': 0.211, 'oracle_percent': 0.015, 'oracle_rank': 7.68, 'prompt_length': 1897.905, 'perplexity': 1.047, 'retrieval_consistency': 1.952, 'syntax_error_percent': 0.25, 'semantic_error_percent': 0.381}
    },
    'DS1000': {
        'top_1': {'pass@1': 0.235, 'ret_recall': 0.043, 'oracle_percent': 0.108, 'oracle_rank': 1.0, 'prompt_length': 1045.637, 'perplexity': 1.044, 'retrieval_consistency': 0.376, 'syntax_error_percent': 0.019, 'semantic_error_percent': 0.42},
        'top_5': {'pass@1': 0.323, 'ret_recall': 0.117, 'oracle_percent': 0.054, 'oracle_rank': 2.31, 'prompt_length': 3139.516, 'perplexity': 1.039, 'retrieval_consistency': 0.924, 'syntax_error_percent': 0.019, 'semantic_error_percent': 0.369},
        'top_10': {'pass@1': 0.323, 'ret_recall': 0.181, 'oracle_percent': 0.041, 'oracle_rank': 4.262, 'prompt_length': 5508.363, 'perplexity': 1.039, 'retrieval_consistency': 1.624, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.325},
        'top_15': {'pass@1': 0.295, 'ret_recall': 0.21, 'oracle_percent': 0.033, 'oracle_rank': 5.692, 'prompt_length': 6281.465, 'perplexity': 1.043, 'retrieval_consistency': 1.949, 'syntax_error_percent': 0.019, 'semantic_error_percent': 0.331},
        'top_20': {'pass@1': 0.312, 'ret_recall': 0.229, 'oracle_percent': 0.027, 'oracle_rank': 6.536, 'prompt_length': 6745.025, 'perplexity': 1.047, 'retrieval_consistency': 2.197, 'syntax_error_percent': 0.025, 'semantic_error_percent': 0.293}
    },
    'pandas_numpy_eval': {
        'top_1': {'pass@1': 0.629, 'ret_recall': 0.099, 'oracle_percent': 0.12, 'oracle_rank': 1.0, 'prompt_length': 507.605, 'perplexity': 1.038, 'retrieval_consistency': 0.377, 'syntax_error_percent': 0.006, 'semantic_error_percent': 0.347},
        'top_5': {'pass@1': 0.725, 'ret_recall': 0.222, 'oracle_percent': 0.053, 'oracle_rank': 2.068, 'prompt_length': 1805.76, 'perplexity': 1.037, 'retrieval_consistency': 0.928, 'syntax_error_percent': 0.012, 'semantic_error_percent': 0.287},
        'top_10': {'pass@1': 0.737, 'ret_recall': 0.294, 'oracle_percent': 0.036, 'oracle_rank': 3.533, 'prompt_length': 3438.928, 'perplexity': 1.039, 'retrieval_consistency': 1.353, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.281},
        'top_15': {'pass@1': 0.701, 'ret_recall': 0.357, 'oracle_percent': 0.029, 'oracle_rank': 5.0, 'prompt_length': 4953.91, 'perplexity': 1.037, 'retrieval_consistency': 1.647, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.263},
        'top_20': {'pass@1': 0.713, 'ret_recall': 0.395, 'oracle_percent': 0.024, 'oracle_rank': 6.213, 'prompt_length': 6499.341, 'perplexity': 1.039, 'retrieval_consistency': 1.874, 'syntax_error_percent': 0.018, 'semantic_error_percent': 0.269}
    }
}


qa_ret_doc_selection_topk_llama_n_1 = {
    'NQ': {
        "top_1": {'em': 0.11, 'f1': 0.246, 'prec': 0.209, 'recall': 0.524, 'ret_recall': 0.472, 'oracle_percent': 0.472, 'oracle_rank': 1.0, 'prompt_length': 282.027, 'perplexity': 1.068},
        "top_5": {'em': 0.04, 'f1': 0.19, 'prec': 0.136, 'recall': 0.64, 'ret_recall': 0.735, 'oracle_percent': 0.306, 'oracle_rank': 1.687, 'prompt_length': 921.669, 'perplexity': 1.068},
        "top_10": {'em': 0.064, 'f1': 0.211, 'prec': 0.16, 'recall': 0.657, 'ret_recall': 0.816, 'oracle_percent': 0.242, 'oracle_rank': 2.277, 'prompt_length': 1720.924, 'perplexity': 1.076},
        "top_15": {'em': 0.061, 'f1': 0.209, 'prec': 0.159, 'recall': 0.641, 'ret_recall': 0.847, 'oracle_percent': 0.21, 'oracle_rank': 2.666, 'prompt_length': 2527.202, 'perplexity': 1.076},
        "top_20": {'em': 0.011, 'f1': 0.104, 'prec': 0.068, 'recall': 0.567, 'ret_recall': 0.871, 'oracle_percent': 0.19, 'oracle_rank': 3.075, 'prompt_length': 3332.663, 'perplexity': 1.083},
    },
    "TriviaQA": {
        "top_1": {'em': 0.298, 'f1': 0.444, 'prec': 0.39, 'recall': 0.775, 'ret_recall': 0.627, 'oracle_percent': 0.627, 'oracle_rank': 1.0, 'prompt_length': 293.966, 'perplexity': 1.063},
        "top_5": {'em': 0.146, 'f1': 0.331, 'prec': 0.259, 'recall': 0.838, 'ret_recall': 0.871, 'oracle_percent': 0.511, 'oracle_rank': 1.5, 'prompt_length': 946.482, 'perplexity': 1.06},
        "top_10": {'em': 0.202, 'f1': 0.378, 'prec': 0.309, 'recall': 0.861, 'ret_recall': 0.917, 'oracle_percent': 0.443, 'oracle_rank': 1.8, 'prompt_length': 1759.983, 'perplexity': 1.068},
        "top_15": {'em': 0.127, 'f1': 0.314, 'prec': 0.241, 'recall': 0.84, 'ret_recall': 0.937, 'oracle_percent': 0.401, 'oracle_rank': 2.03, 'prompt_length': 2579.584, 'perplexity': 1.073},
        "top_20": {'em': 0.014, 'f1': 0.142, 'prec': 0.089, 'recall': 0.743, 'ret_recall': 0.952, 'oracle_percent': 0.375, 'oracle_rank': 2.271, 'prompt_length': 3398.936, 'perplexity': 1.084}
    },
    "hotpotQA": {
        "top_1": {'em': 0.123, 'f1': 0.242, 'prec': 0.216, 'recall': 0.457, 'ret_recall': 0.351, 'oracle_percent': 0.703, 'oracle_rank': 1.0, 'prompt_length': 247.024, 'perplexity': 1.062},
        "top_5": {'em': 0.077, 'f1': 0.199, 'prec': 0.159, 'recall': 0.521, 'ret_recall': 0.582, 'oracle_percent': 0.233, 'oracle_rank': 1.7, 'prompt_length': 656.491, 'perplexity': 1.06},
        "top_10": {'em': 0.046, 'f1': 0.169, 'prec': 0.125, 'recall': 0.536, 'ret_recall': 0.639, 'oracle_percent': 0.128, 'oracle_rank': 2.225, 'prompt_length': 1150.41, 'perplexity': 1.06},
        "top_15": {'em': 0.036, 'f1': 0.162, 'prec': 0.12, 'recall': 0.523, 'ret_recall': 0.666, 'oracle_percent': 0.089, 'oracle_rank': 2.665, 'prompt_length': 1648.619, 'perplexity': 1.065},
        "top_20": {'em': 0.03, 'f1': 0.145, 'prec': 0.105, 'recall': 0.498, 'ret_recall': 0.688, 'oracle_percent': 0.069, 'oracle_rank': 3.172, 'prompt_length': 2144.122, 'perplexity': 1.069}
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
        'top_1': {'em': 0.254, 'f1': 0.391, 'prec': 0.381, 'recall': 0.5, 'ret_recall': 0.472, 'oracle_percent': 0.472, 'oracle_rank': 1.0, 'prompt_length': 226.675, 'perplexity': 1.04},
        "top_5": {'em': 0.327, 'f1': 0.468, 'prec': 0.454, 'recall': 0.603},
        "top_10": {'em': 0.345, 'f1': 0.491, 'prec': 0.476, 'recall': 0.622},
        "top_15": {'em': 0.345, 'f1': 0.492, 'prec': 0.478, 'recall': 0.624},
        'top_20': {'em': 0.349, 'f1': 0.497, 'prec': 0.483, 'recall': 0.629, 'ret_recall': 0.871, 'oracle_percent': 0.19, 'oracle_rank': 3.075, 'prompt_length': 2833.01, 'perplexity': 1.042},
        'top_40': {'em': 0.346, 'f1': 0.496, 'prec': 0.482, 'recall': 0.635, 'ret_recall': 0.911, 'oracle_percent': 0.147, 'oracle_rank': 4.238, 'prompt_length': 5577.404, 'perplexity': 1.048},
        'top_60': {'em': 0.344, 'f1': 0.492, 'prec': 0.476, 'recall': 0.638, 'ret_recall': 0.932, 'oracle_percent': 0.127, 'oracle_rank': 5.244, 'prompt_length': 8323.228, 'perplexity': 1.052},
        'top_80': {'em': 0.337, 'f1': 0.487, 'prec': 0.471, 'recall': 0.633, 'ret_recall': 0.944, 'oracle_percent': 0.114, 'oracle_rank': 6.049, 'prompt_length': 11070.615, 'perplexity': 1.06}
    },
    'TriviaQA': {
        'top_1': {'em': 0.592, 'f1': 0.679, 'prec': 0.658, 'recall': 0.767, 'ret_recall': 0.627, 'oracle_percent': 0.627, 'oracle_rank': 1.0, 'prompt_length': 236.653, 'perplexity': 1.028},
        "top_5": {'em': 0.645, 'f1': 0.727, 'prec': 0.706, 'recall': 0.812},
        "top_10": {'em': 0.671, 'f1': 0.752, 'prec': 0.729, 'recall': 0.837},
        "top_15": {'em': 0.673, 'f1': 0.757, 'prec': 0.733, 'recall': 0.843},
        'top_20': {'em': 0.677, 'f1': 0.758, 'prec': 0.735, 'recall': 0.84, 'ret_recall': 0.952, 'oracle_percent': 0.375, 'oracle_rank': 2.271, 'prompt_length': 2904.057, 'perplexity': 1.025},
        'top_40': {'em': 0.697, 'f1': 0.774, 'prec': 0.75, 'recall': 0.857, 'ret_recall': 0.975, 'oracle_percent': 0.316, 'oracle_rank': 2.883, 'prompt_length': 5714.368, 'perplexity': 1.029},
        'top_60': {'em': 0.7, 'f1': 0.775, 'prec': 0.752, 'recall': 0.856, 'ret_recall': 0.984, 'oracle_percent': 0.285, 'oracle_rank': 3.321, 'prompt_length': 8529.492, 'perplexity': 1.036},
        'top_80': {'em': 0.683, 'f1': 0.763, 'prec': 0.739, 'recall': 0.855, 'ret_recall': 0.986, 'oracle_percent': 0.263, 'oracle_rank': 3.459, 'prompt_length': 11346.743, 'perplexity': 1.043}
    },
    'hotpotQA': {
        'top_1': {'em': 0.234, 'f1': 0.355, 'prec': 0.361, 'recall': 0.408, 'ret_recall': 0.351, 'oracle_percent': 0.703, 'oracle_rank': 1.0, 'prompt_length': 195.152, 'perplexity': 1.041},
        "top_5": {'em': 0.2785, 'f1': 0.408, 'prec': 0.416, 'recall': 0.462},
        "top_10": {'em': 0.295, 'f1': 0.421, 'prec': 0.426, 'recall': 0.475},
        "top_15": {'em': 0.305, 'f1': 0.433, 'prec': 0.439, 'recall': 0.485},
        'top_20': {'em': 0.305, 'f1': 0.431, 'prec': 0.439, 'recall': 0.483, 'ret_recall': 0.688, 'oracle_percent': 0.069, 'oracle_rank': 3.172, 'prompt_length': 1800.811, 'perplexity': 1.039},
        'top_40': {'em': 0.323, 'f1': 0.453, 'prec': 0.461, 'recall': 0.506, 'ret_recall': 0.732, 'oracle_percent': 0.037, 'oracle_rank': 4.764, 'prompt_length': 3459.514, 'perplexity': 1.043},
        'top_60': {'em': 0.311, 'f1': 0.442, 'prec': 0.45, 'recall': 0.494, 'ret_recall': 0.751, 'oracle_percent': 0.025, 'oracle_rank': 5.917, 'prompt_length': 5106.305, 'perplexity': 1.049},
        'top_80': {'em': 0.305, 'f1': 0.441, 'prec': 0.449, 'recall': 0.495, 'ret_recall': 0.77, 'oracle_percent': 0.019, 'oracle_rank': 7.506, 'prompt_length': 6743.218, 'perplexity': 1.054}
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
            'oracle': {'pass@1': 0.381, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.589, 'prompt_length': 820.869, 'perplexity': 1.038, 'retrieval_consistency': 0.94, 'syntax_error_percent': 0.167, 'semantic_error_percent': 0.25},
            'pl_500': {'pass@1': 0.333, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.226, 'prompt_length': 503.619},
            'pl_2000': {'pass@1': 0.357, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.552, 'prompt_length': 2010.036},
            'pl_4000': {'pass@1': 0.357, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.589, 'prompt_length': 3917.75},
            'pl_8000': {'pass@1': 0.321, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.589, 'prompt_length': 7127.048}
        },
        'random': {
            'random': {'pass@1': 0.262, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 402.667, 'perplexity': 1.061, 'retrieval_consistency': 0.024, 'syntax_error_percent': 0.262, 'semantic_error_percent': 0.381},
            'pl_500': {'pass@1': 0.321, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 491.19},
            'pl_2000': {'pass@1': 0.321, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 1997.595},
            'pl_4000': {'pass@1': 0.345, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 3990.845},
            'pl_8000': {'pass@1': 0.333, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 7983.607}
        },
        'irrelevant_diff': {
            'irrelevant_diff': {'pass@1': 0.321, 'prompt_length': 814.833, 'perplexity': 1.047, 'syntax_error_percent': 0.369, 'semantic_error_percent': 0.405},
            'pl_500': {'pass@1': 0.274, 'prompt_length': 511.476},
            'pl_2000': {'pass@1': 0.31, 'prompt_length': 2036.19},
            'pl_4000': {'pass@1': 0.333, 'prompt_length': 3966.048},
            'pl_8000': {'pass@1': 0.345, 'prompt_length': 7197.583}
        },
        'irrelevant_dummy': {
            'irrelevant_dummy': {'pass@1': 0.286, 'prompt_length': 814.321, 'perplexity': 1.048, 'syntax_error_percent': 0.381, 'semantic_error_percent': 0.417},
            'pl_500': {'pass@1': 0.274, 'prompt_length': 512.143},
            'pl_2000': {'pass@1': 0.345, 'prompt_length': 2037.429},
            'pl_4000': {'pass@1': 0.286, 'prompt_length': 3967.869},
            'pl_8000': {'pass@1': 0.321, 'prompt_length': 7195.548}
        }
    }
}

code_pl_analysis_llama_n_1 = {}



qa_pl_analysis_gpt_n_1 = {
    'NQ': {
        'none': {'em': 0.247, 'f1': 0.403, 'prec': 0.381, 'recall': 0.603, 'prompt_length': 64.483, 'perplexity': 1.064},
        'oracle': {
            'oracle': {'em': 0.519, 'f1': 0.676, 'prec': 0.657, 'recall': 0.82, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 228.052},
            'pl_500': {'em': 0.522, 'f1': 0.68, 'prec': 0.659, 'recall': 0.826, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 488.079},
            'pl_2000': {'em': 0.532, 'f1': 0.689, 'prec': 0.67, 'recall': 0.825, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 1981.419, 'perplexity': 1.026},
            'pl_4000': {'em': 0.535, 'f1': 0.692, 'prec': 0.674, 'recall': 0.828, 'ret_recall': 1.0, 'oracle_percent': 1.0, 'oracle_rank': 1.0, 'prompt_length': 3981.055}
        },
        # 'distracting': {
        #     'distracting': {'em': 0.077, 'f1': 0.189, 'prec': 0.195, 'recall': 0.257, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 225.648, 'perplexity': 1.05},
        #     'pl_2000': {'em': 0.067, 'f1': 0.177, 'prec': 0.184, 'recall': 0.238, 'ret_recall': 0.0, 'oracle_percent': 0.0, 'prompt_length': 1982.787, 'perplexity': 1.048},
        #     'pl_4000': {},
        # },
        'retrieved': {
            'retrieved': {'em': 0.345, 'f1': 0.491, 'prec': 0.476, 'recall': 0.622},
            'pl_2000': {'em': 0.35, 'f1': 0.494, 'prec': 0.48, 'recall': 0.621, 'ret_recall': 0.816, 'oracle_percent': 0.268, 'oracle_rank': 2.277, 'prompt_length': 1982.521, 'perplexity': 1.037},
            'pl_4000': {},
        },  # top10 for QA, top5 for Code
        'random': {
            'random': {'em': 0.164, 'f1': 0.265, 'prec': 0.263, 'recall': 0.355, 'ret_recall': 0.003, 'oracle_percent': 0.003, 'oracle_rank': 1.0, 'prompt_length': 230.421},
            'pl_500': {'em': 0.149, 'f1': 0.248, 'prec': 0.242, 'recall': 0.345, 'ret_recall': 0.011, 'oracle_percent': 0.004, 'oracle_rank': 1.619, 'prompt_length': 496.366},
            'pl_2000': {'em': 0.19, 'f1': 0.301, 'prec': 0.303, 'recall': 0.387, 'ret_recall': 0.035, 'oracle_percent': 0.004, 'oracle_rank': 6.507, 'prompt_length': 1976.963, 'perplexity': 1.083},
            'pl_4000': {'em': 0.182, 'f1': 0.292, 'prec': 0.293, 'recall': 0.379, 'ret_recall': 0.059, 'oracle_percent': 0.003, 'oracle_rank': 12.205, 'prompt_length': 3976.886}
        },
        'irrelevant_diff': {
            'irrelevant_diff': {'em': 0.142, 'f1': 0.2265, 'prec': 0.224, 'recall': 0.307, 'prompt_length': 228.579},
            'pl_500': {'em': 0.125, 'f1': 0.21, 'prec': 0.204, 'recall': 0.303, 'prompt_length': 489.673},
            'pl_2000': {'em': 0.161, 'f1': 0.265, 'prec': 0.261, 'recall': 0.362, 'prompt_length': 1987.573, 'perplexity': 1.094},
            'pl_4000': {'em': 0.156, 'f1': 0.258, 'prec': 0.257, 'recall': 0.35, 'prompt_length': 3999.585}
        },
        'irrelevant_dummy': {
            'irrelevant_dummy': {'em': 0.12, 'f1': 0.198, 'prec': 0.1875, 'recall': 0.288, 'prompt_length': 228.262},
            'pl_500': {'em': 0.111, 'f1': 0.197, 'prec': 0.186, 'recall': 0.296, 'prompt_length': 488.71},
            'pl_2000': {'em': 0.154, 'f1': 0.254, 'prec': 0.246, 'recall': 0.35, 'prompt_length': 1984.417, 'perplexity': 1.087},
            'pl_4000': {'em': 0.159, 'f1': 0.265, 'prec': 0.258, 'recall': 0.367, 'prompt_length': 3987.259}
        },
        'ellipsis': {
            'pl_500': {'em': 0.239, 'f1': 0.371, 'prec': 0.366, 'recall': 0.489, 'prompt_length': 490.941, 'perplexity': 1.073}
        },   # pad potential doc with ellipsis
        'ellipsis_and_pretend': {

        },   # pad doc with ellipsis and pretend this to be a document that contains information
        # 'self_pretend': {
        #
        # },   # ask llm to pretend that there are oracle document
        'self_pad': {

        },   # pad none with ellipsis
        'self_generate': {'em': 0.007, 'f1': 0.177, 'prec': 0.11, 'recall': 0.648, 'prompt_length': 88.483, 'perplexity': 1.129},   # let llm generate documents, and then answer the question
        # todo: need more exps, keep same semantic, add prompt lengths (need to pay attention to the position of the retrieved docs)
        # todo: is the random information itself can do this or the prompt length is the key?
        # todo: more prompt length can help LLM revoke its own knowledge, use pretend and ...
        # todo: get all results -> then discuss
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
