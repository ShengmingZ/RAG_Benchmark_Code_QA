import ast
import os.path
import platform
import sys
import json
import re
import numpy as np
from typing import List
import tempfile
import subprocess
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/RAG_Benchmark_Code_QA'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/RAG_Benchmark_Code_QA'
sys.path.insert(0, root_path)
from dataset_utils.conala_utils import ConalaLoader
from generator_deprecated.generate_utils import generate_config, get_docs_tokens
from dataset_utils.DS1000_utils import DS1000Loader
from data.DS1000.ds1000 import DS1000Dataset
from dataset_utils.pandas_numpy_eval_utils import PandasNumpyEvalLoader
from dataset_utils.NQ_TriviaQA_utils import NQTriviaQAUtils
from dataset_utils.hotpotQA_utils import HotpotQAUtils
from retriever.retriever_utils import ret_eval_by_doc_keys
from data_processing.analyze_result import analyze_results_for_code


def conala_result_process(prompt_type, output, output_before=None):
    pred = output
    if prompt_type == 'self-refine' and '```' not in output and '<code>' not in output: pred = output_before   # if no refine in self-refine, just use output before
    pred = pred.replace('</s>', '').replace('```python', '```')
    try: pred = pred.split('Potential documents')[0]
    except: ...
    if prompt_type in ['least_to_most']:
        try: pred = pred.rsplit('```', 1)[0].rsplit('```', 1)[1]
        except: ...
    try: pred = pred.split('<code>')[1].split('</code>')[0]
    except: ...
    try: pred = pred.split('```')[1].split('```')[0]
    except: ...
    # only keep one code line
    pred_lines = [line for line in pred.split('\n') if line != '' and not line.startswith('#') and not line.startswith('    #')]
    try:
        if pred_lines[-1].startswith('print'): pred = pred_lines[-2]
        else: pred = pred_lines[-1]
    except: ...
    return pred


def DS1000_result_process(prompt_type, output, code_prompt, output_before=None):
    pred = output
    if prompt_type == 'self-refine' and not '```' in output and not '<code>' in output: pred = output_before
    pred = pred.replace('</s>', '').replace('```python', '```')
    try: pred = pred.split('Potential documents')[0]
    except: ...
    if prompt_type in ['least_to_most']:
        try: pred = pred.rsplit('```', 1)[0].rsplit('```', 1)[1]
        except: ...
    try: pred = pred.split('BEGIN SOLUTION')[1]
    except: ...
    try: pred = pred.split('END SOLUTION')[0]
    except: ...
    try: pred = pred.split('```', 1)[1].split('```', 1)[0]
    except: ...
    try: pred = pred.split('<code>')[1].split('</code>')[0]
    except: ...

    prompt_lines = code_prompt.split('\n')
    prompt_lines = [line for line in prompt_lines if line != '' and not line.startswith('#') and not line.startswith('    #')]
    # print(prompt_lines)
    pred_lines = pred.split('\n')
    pred_lines = [line for line in pred_lines if line != '' and not line.startswith('#') and not line.startswith('    #')]
    # print(pred_lines)

    preload_variables = []  # get pre-defined variables in code prompt
    for prompt_line in prompt_lines:
        if ' = ' in prompt_line: preload_variables.extend([var.replace(' ', '') for var in prompt_line.split('=')[0].split(',')])
        if 'BEGIN SOLUTION' in prompt_line: break
    # if model output full code snippet, need to remove duplicated ones
    # and sometimes LLM would change the definition of preload variables e.g.: softmax_output = load_data() -> softmax_output = torch.tensor([[0.2, 0.1, 0.7], ...
    _pred_lines = []
    for pred_line in pred_lines:
        is_same = False
        # remove dup
        for prompt_line in prompt_lines:
            if prompt_line.replace(' ', '') == pred_line.replace(' ', ''):
                for var in preload_variables:
                    if ' = ' in prompt_line and var in prompt_line.split('=')[0]: preload_variables.remove(var)  # if not change the defi...
                is_same = True
        if not is_same: _pred_lines.append(pred_line)
    pred = '\n'.join(_pred_lines)

    # if model have output full code snippet and change the definition of preload variables
    if len(pred_lines) - len(_pred_lines) >= 2 and len(preload_variables) > 0:
        _pred_lines = []
        pred_lines = pred.split('\n')
        for pred_line in pred_lines:
            is_same = False
            for var in preload_variables:
                if pred_line.startswith(f'{var} ='):
                    preload_variables.remove(var)
                    is_same = True
            if not is_same: _pred_lines.append(pred_line)
        pred = '\n'.join(_pred_lines)
    return pred


def pandas_numpy_eval_result_process(prompt_type, output, code_prompt, output_before=None):
    # first extract code
    pred = output
    if prompt_type == 'self-refine' and not '```' in output and not '<code>' in output: pred = output_before
    if pred.startswith(' '): pred = pred[1:]
    pred = pred.replace('</s>', '').replace('```python', '```')
    try: pred = pred.split('Potential documents')[0]
    except: ...
    if prompt_type in ['least_to_most']:
        try: pred = pred.rsplit('```', 1)[0].rsplit('```', 1)[1]
        except: ...
    try: pred = pred.split('<code>')[1]
    except: ...
    try: pred = pred.split('</code>')[0]
    except: ...
    try: pred = pred.split('```', 1)[1].split('```', 1)[0]
    except: ...
    try: pred = pred.split('# Example usage')[0]
    except: ...
    try: pred = pred.split('[out]')[0]
    except: ...
    pred = pred.replace('`', '')
    # clean code
    prompt_lines = code_prompt.split('\n')
    prompt_lines = [line for line in prompt_lines if line != '' and not line.startswith('#') and not line.startswith('    #')]
    # print(prompt_lines)
    pred_lines = pred.split('\n')
    pred_lines = [line for line in pred_lines if line != '' and not line.startswith('#') and not line.startswith('    #')]
    # print(pred_lines)
    # remove dup lines
    _pred_lines = []
    for pred_line in pred_lines:
        is_same = False
        for prompt_line in prompt_lines:
            if prompt_line.replace(' ', '') == pred_line.replace(' ', ''): is_same = True
        if not is_same: _pred_lines.append(pred_line)
    pred_lines = _pred_lines
    # remove partial dup in the last line
    try:
        if pred_lines[0].startswith(prompt_lines[-1]): pred_lines[0] = pred_lines[0].replace(prompt_lines[-1], '')
    except: ...
    # add indent for return
    try:
        if prompt_lines[-1] != '    ' and pred_lines[0].startswith('return'): pred_lines[0] = '    ' + pred_lines[0]
    except: ...
    # add intent for all pred lines
    try:
        if 'def' in code_prompt and prompt_lines[-1] != '    ' and not pred_lines[0].startswith('    '): pred_lines = ['    ' + line for line in pred_lines]
    except: ...
    # add return for function
    try:
        if 'def' in code_prompt and 'return' not in pred:
            var = pred_lines[-1].split(' = ')[0].replace(' ', '')
            pred_lines.append(f'    return {var}')
    except: ...
    pred = '\n'.join(pred_lines)
    return pred


def process_gene_results(qs_id, dataset, outputs, prompt_type=None, code_prompt=None, outputs_before=None):
    preds = []
    if dataset == 'conala':
        for idx, output in enumerate(outputs):
            if prompt_type == 'self-refine':
                pred = conala_result_process(prompt_type, output, outputs_before[idx])
            else:
                pred = conala_result_process(prompt_type, output)
            preds.append(pred)

    elif dataset == 'DS1000':
        for idx, output in enumerate(outputs):
            if prompt_type == 'self-refine':
                pred = DS1000_result_process(prompt_type, output, code_prompt, outputs_before[idx])
            else:
                pred = DS1000_result_process(prompt_type, output, code_prompt)
            preds.append(pred)

    elif dataset == 'pandas_numpy_eval':
        for idx, output in enumerate(outputs):
            if prompt_type == 'self-refine':
                pred = pandas_numpy_eval_result_process(prompt_type, output, code_prompt, outputs_before[idx])
            else:
                pred = pandas_numpy_eval_result_process(prompt_type, output, code_prompt)
            preds.append(pred)

    elif dataset == 'NQ' or dataset == 'TriviaQA' or dataset == 'hotpotQA':
        for idx, output in enumerate(outputs):
            pred = output
            # if prompt_type == 'RaR':
            #     try: pred = pred.split('Answer:\n')[1]
            #     except: ...
            #     try: pred = pred.split('the answer')[1]
            #     except: ...
            preds.append(pred)

    else:
        raise Exception('Not Implemented')

    return preds


def process_outputs_for_self_consistency(outputs):
    outputs_dict = dict()
    for output in outputs:
        if output in outputs_dict.keys():
            outputs_dict[output] += 1
        else:
            outputs_dict[output] = 1
    most_output = sorted(outputs_dict.items(), key=lambda item: item[1], reverse=True)[0][0]
    return most_output


# def parsing_ds1000_by_llm(generated_code: str, existing_code: str):
#     from llms.OpenAIProvider import OpenAIProvider
#     llm_provider = OpenAIProvider(organization='openai',
#                                   model='gpt-4.1',
#                                   temperature=0,
#                                   max_tokens=1000,
#                                   is_async=False,
#                                   stop=None)
#     prompt_template = """The original code has [insert] placeholders. The generated code should keep all original lines exactly the same and only replace [insert] with actual code.
#
# Fix any parts where the generated code changed the original lines. Keep the same functionality.
#
#
# original code:
# {original_code}
#
# generated code:
# {generated_code}
#
# Output the corrected code, wrapped in <code> and </code>."""
#     prompt = prompt_template.format(original_code=existing_code, generated_code=generated_code)
#     # print(prompt)
#     llm_response = llm_provider.generate([{'role': 'user', 'content': prompt}], include_logits=False)
#     try:
#         llm_response = llm_response.split('<code>')[1].split('</code>')[0]
#     except: print(llm_response)
#     try:
#         llm_response = llm_response.split('```python')[1].split('```')[0]
#     except: print(llm_response)
#     return llm_response
#
# if __name__ == '__main__':
#     generated_code = """import pandas as pd
#
# df=pd.DataFrame(data=[[1,1,2,5],[1,3,4,1],[4,1,2,5],[5,1,4,9],[1,1,2,5]],columns=['val', 'col1','col2','3col'])
#
# duplicate_bool = df.duplicated(subset=['col1','col2', '3col'], keep='first')
# duplicate = df.loc[duplicate_bool == True]
# # Adding a column referring to the index of the first duplicate
# duplicate['index_original'] = df[duplicate_bool].index[0]
# print(duplicate)"""
#     original_code = """import pandas as pd
#
# df=pd.DataFrame(data=[[1,1,2,5],[1,3,4,1],[4,1,2,5],[5,1,4,9],[1,1,2,5]],columns=['val', 'col1','col2','3col'])
# [insert]
# print(result)"""
#     print(parsing_ds1000_by_llm(generated_code, existing_code=original_code))



def process_ds1000_outputs(pid: str, outputs: List[str], existing_code: str):
    """
    replace existing code in DS1000, because the test code is different from code prompt
    :param outputs:
    :param existing_code:
    :return:
    """
    processed_outputs = []
    for output in outputs:
        output_statements = output.split('\n')
        for code_statement in existing_code.split('\n'):
            if code_statement not in ['<code>', '</code>', 'BEGIN SOLUTION', 'END SOLUTION', '[insert]', 'runnable code', 'Runnable code', 'corrected, runnable code', '']:
                # if regex matching cannot solve, use LLM as parser
                if code_statement not in output_statements:
                    print(f'***********parsing error, LLM fail to keep the existing code exactly the same in {pid}: {[code_statement]}')
                else: output_statements.remove(code_statement)
        processed_outputs.append('\n'.join(output_statements))
        # else:
        #     parsed_code = parsing_ds1000_by_llm(generated_code=output, existing_code=existing_code)
        #     output_statements = parsed_code.split('\n')
        #     for code_statement in existing_code.split('\n'):
        #         if code_statement not in ['<code>', '</code>', 'BEGIN SOLUTION', 'END SOLUTION', '[insert]',
        #                                   'runnable code', 'Runnable code', 'corrected, runnable code', '']:
        #             if code_statement not in output_statements:
        #                 print(f'***********parsing error, LLM fail to keep the existing code exactly the same in {pid}: {[code_statement]}')
        #                 # if regex matching cannot solve, use LLM as parser
        #             else:
        #                 output_statements.remove(code_statement)
        #     processed_outputs.append('\n'.join(output_statements))
    return processed_outputs




def check_python_code_ruff(code_string):
    """
    Check Python code string using Ruff for errors only (ignoring warnings)

    Args:
        code_string (str): Python code to check

    Returns:
        bool: True if no errors found, False if errors detected
    """
    common_imports = {
        'pd', 'pandas',  # pandas
        'np', 'numpy',  # numpy
        'tf', 'tensorflow',  # tensorflow
        'torch', 'pytorch',  # pytorch
        'sk', 'sklearn',  # scikit-learn
        'sns', 'seaborn',  # seaborn
        'json', 'pickle',  # serialization
        'os', 'sys', 'pathlib',  # system
        'time', 'datetime',  # time
        're', 'regex',  # regex
        'math', 'statistics',  # math
        'collections', 'itertools',  # collections
        'typing', 'dataclasses',  # typing
        # todo: weird function name for conala
        'array_equal', 'indexOf', 'parse_datetime_string', 'set_warn_always', 'find_duplicate', 'file_exists', '_record_count', '_remove_dups', 'get', 'write_bytes',
        'makePickle', 'formatweekday', 'intersection', 'split', 'timedelta', 'today', 'ensure_decoded', 'encode_base64', 'to_list', 'atoi', '_record_count',
        '_follow_symlinks', 'split_outside_bracket', '_sort_dump_data_by', 'itervalues', 'setvar', 'find', 'unhex', '_delimited_splitter', 'get_attrs', 'mean', 'utcnow', 'concatenate',
        'unhexlify', 'b64encode', 'dedent', '_munge_whitespace', 'timezone', 'unquote_plus', 'fromisoformat', 'builtins', 'DataFrame', 'Sequence', 'extract_bool_array', 'set_array',
        'eliminate_zeros', '_find_valid_index', '_remove_vertex', 'set_charmap', 'itn', 'previous_friday', 'unquote', 'before_nearest_workday', 'plain', 'countOf', 'drive_files',
        'unescape', 'encode', '_parse_date', 'datestr', 'date_format', 'strftime', '_convert_strls', 'size', 'file_exists', 'contains', 'temp_setattr', 'set_charmap', 'remove_axis',
        'set_char', 'ishex', 'splitattr', 'get_intersection', 'BooleanArray', 'unique_key', 'test', 'reverse_dict', '_normalise_json_ordered', '_list_of_dict_to_arrays', '_sorted',
        'valfilter', '_matrix_vector_product_of_stacks', 'get_lastbday', '_prev_opening_time', 'before_nearest_workday', 'Int2AP', 'add_object_type_line', '_log_normalize',
        '_prepare_categoricals', '_replace_nans', '_convert_strls', '_shape_common', 'samestat', '_datetime_to_stata_elapsed_vec', 'previous_workday', 'duplicated', '_filter',
        'fast_unique_multiple', 'strip_newsgroup_footer', 'strip_newsgroup_quoting', '_checkpoint_exists', '_flatten_dims_0_and_1', '_lookup_reduction', 'replace_list', 'filter_sources',
        'newer_pairwise', 'targets', 'multicolumn', '_get_json_content_from_openml_api', '_find_lteq', 'random', 'a', 'b', 'abracadabra', 'newFile', 'newFileBytes', '_create_block_3_diagonal_matrix',
        'd', '_multi_dot_three', 'subspace_angles', '_matrix_vector_product_of_stacks', '_remove_zero_rows', 'indices_to_mask', 'asof_locs', '_all_string_prefixes', 'validate_strlist',
        'isalnum', 'nearest_workday', 'after_nearest_workday', '_format_multicolumn', '_Flatten', 'df', 's', '_abc_registry_clear', 'clean', 'prune', 'difference', 'l1', 'l2', 'stopwords',
        'remove_vertex', 'unescaped', 'previous_workday', 'StataStrLWriter', 'matmul', '_remove_zero_rows', 'remove', 'match_extensions', 'what', '_guess_quote_and_delimiter', 'splitattr', 'as_hex',
        'resetwarnings', 'warning', 'exception', '_missing_warn', 'warn_explicit', 'set_warn_always', 'resetwarnings', 'match', 'symmetric_difference', 'tolil', '_format_multirow', '_Flatten', 'floor',
        'load_data', 'sklearn_model', 'data', 'float16', 'result', 'x', 'y', 'index', 'columns', 'sparse_csr_matrix_ops', 'solve_ivp', 'sparse', 'X', 'train_test_split', 'data_matrix', 'X_train', 'y_train',
        'GradientBoostingClassifier', 'my_map_func', 'NormalDistro', 'train_size', 'features_dataframe', '_reshape_2D', '_to_tensor', 'tensor', 'standard_scale', 'make_np', 'predict',
        'sparse_matrix_sparse_mat_mul', 'RandomForestRegressor', 'X_test', 'dataset', 'scaled', 'fitted_model', 'W', 't', 'exp', 'df_a', 'df_b', '_unpack', 'sciopt', 'e', 'pmin', 'pmax',
        'scipy', 'points', 'extraPoints', 'model', 'LinearSVC', 'vectorizer', 'preprocessing', 'SelectFromModel', 'clf', 'TfidfVectorizer', 'cv', 'logreg', 'km', 'hid_dim', 'softmax_output',
        'rotate_around', 'make_friedman1', 'scores', 'predicted_t_scaled', '_export_model_variables', 'make_friedman2', 'datasets', 'GradientBoostingClassifier', 'clean_data', 'time_span', 'N0', 'integrate',
        'transform_output', 'df_origin', 'new_data', 'C', 'D', 'rotate_deg_around', 'load_target', 'features_dataframe', 'regression_model', 'strs', 'example_df', '_find_numeric_cols', 'DataArray',
        'percentile', 'self', 'r', 'c', 'im', 'rankdata', 'N', 'orthogonal_procrustes', '_sparse_manhattan', '_quadratic_assignment_2opt', 'assertNDArrayNear', 'contiguous_regions',
        'assertNDArrayNear', 'is_evenly_distributed_thresholds', 'f_classif', 'query', 'apply_2d', '_feature_to_dtype', '_python_apply_general', 'split_training_and_validation_data', '_do_convert_categoricals',
        '_sanitize_ndim', '_convert_to_tensors_or_sparse_tensors', 'stacked_matmul', '_cast_tensor_to_floatx', 'convert', 'make_np', '_cast_tensor_to_floatx', 'sanitize_masked_array', 'data_rot', 'suppmach',
        'x_test', 'suppmach', 'X_train_num', 'car', 'values', 'names', 'times', 'regressor', 'insert', 'column_names', 'cosine_similarity', 'get_term_frequency_inverse_data_frequency', 'example_df',
        'example_dict', 'corr', 'thresh', 'post', 'distance', 'fill_zeros', 'factorize', 'nonzero', 'A', 'img', 'threshold', 'dN1_dt_simple', 'solve_ivp', 'tfidf', 'queries', 'objective', 'pad_width',
        'points1', 'points2', 'features', 'diagonal', 'condensed_dist_matrix', 'joblib', 'something', 'array', 'nlargest', 'einsum', 'search', 'where', 'ensure_string_array',
        'ensure_float', 'remove_na_arraylike', 'nanargmin', 'vec', 'can_hold_element', 'z_score', 'isnaobj', 'maybe_fill', 'nanops', 'Normalize', '_try_cast',
        'is_inferred_bool_dtype', 'condition', 'axis_slice', 'get_geometry', 'max_len', 'z', 'target_series', 'source_series', '_sort_tuples', 'arr', 'make_sparse',
        'downcast_intp_index', '_from_backing_data', 'can_hold_element', '_from_backing_data', 'find_repeats', 'is_empty_indexer', 'is_sparse', 'is_scipy_sparse',
        'remove_na_arraylike', 'maybe_cast_to_integer_array', 'Index', 'IndexLabel', 'astype_nansafe', 'maybe_fill', 'unpack_zerodim_and_defer', '_get_dataframe_dtype_counts',
        '_set_noconvert_dtype_columns', '_try_convert_data', 'rfn', '_to_matrix_vectorized', 'rgb_to_hsv', 'NpDtype', 'f', 'emails', '_get_na_values', 'col', '_validate_names', '_check_column_names',
        'loc', 'column', 'value', 'col_names', 'Samples', 's1', 's2', 'col_name', 'df1', 'df2', 'matrix', 'dr', 'stepsize', '_GetDenseDimensions', '_get_dtype_from_nested_lists',
        'is_multilabel', 'pointbiserialr', 'weighted', '_convert_arff_data_dataframe', '_get_dataframe_dtype_counts', 'recursive_fill_fields', '_array_perimeter',
        'is_categorical', 'is_scipy_sparse', 'squeeze', 'compress_rows', 'array_equal', '_formatter', 'encodebytes', 'unquote_plus', 'JSONEncoder', 'get_strcols',
        '_convert_strls', 'endswith', '_cc', 'b32encode', 'strip_math', '_split', 'dedent', 'values_cols', 'pivot_table', '_get_dataframe_dtype_counts', 'start',
        'end', 'what', 'global_variables', '_find_rteq', '_GetPyList', 'sendfile', 'gmean', 'rpartition', 'nti', 'integer', 'set_eng_float_format',
        'Iterable', '_unflatten_first_dim', 'should_use_regex', '_calc_t_stat', 'iterkeys', 'is_re', 'icursor', 'isalpha', 'pop', 'set_charmap', '_remove_vertex',
        '_attr_getter', 'uint32', 'gzip_encode', '_to_str', '_document_frequency', '_get_dataframe_dtype_counts', 'NameListToString', '_remove_redundancy_svd',
        '_remove_zero_rows', '_remove_redundancy_svd', 'bisect_right', 'p', '_with_nonzero_rank', '_convert_strls', 'add_object_type_line', '_log_normalize',
        '_prepare_categoricals', '_calc_oa_lens', '_intersection', 'tolist', '_map_to_integer', 'uniques', 'strip_newsgroup_header', '_munge_whitespace',
        'strip_newsgroup_footer', 'strip_newsgroup_quoting', 'strip_newsgroup_header', 'strip_math', 'getattr_static', 'count', 'value_counts',
        '_get_dataframe_dtype_counts', '_document_frequency', 'argmax', 'align_method_SERIES', 'filter_empty_layer_containers', 'isin', '_get_dataframe_dtype_counts',
        'foobarrrr', 'CountVectorizer'
    }

    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code_string)
            temp_filename = temp_file.name

        try:
            # Run Ruff with error-only selection
            result = subprocess.run([
                'ruff', 'check',
                '--select=E9,F63,F7,F82',  # Error codes only
                temp_filename
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                return True

            stdout_lines = result.stdout.strip().split('\n') if result.stdout else []
            filtered_errors = []
            skip_until_next_error = False

            for line in stdout_lines:
                line = line.strip()
                if not line:
                    continue

                if re.match(r'^[^:]+:\d+:\d+:', line):
                    skip_until_next_error = False  # Reset for new error

                    # Check if this is an F821 error for a common import name
                    # Format: "file:line:col: F821 Undefined name `name`"
                    f821_match = re.search(r'F821.*Undefined name [`\'"](\w+)[`\'"]', line)
                    if f821_match:
                        undefined_name = f821_match.group(1)
                        if undefined_name in common_imports:
                            # Skip this error - it's a common import
                            skip_until_next_error = True
                            continue

                if not skip_until_next_error:
                    filtered_errors.append(line)
                    print(line)


            # Return True if no errors remain after filtering
            if len(filtered_errors) !=0: print(result.stdout)
            return len(filtered_errors) == 0

        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    except subprocess.TimeoutExpired:
        return False
    except FileNotFoundError:
        return False
    except Exception:
        return False




def parsing_for_conala_new(qs_list, model, prompt_method, results):
    _gene_results = list()
    unittest_file = "../data/conala/unittest_docprompting_conala.json"
    unittests = json.load(open(unittest_file, 'r'))
    for idx, result in enumerate(results):
        assert qs_list[idx]['qs_id'] == result['qs_id']
        print(result['qs_id'])
        # first: default parsing for recall and docNum experiments:
        outputs = [result['response'].replace('<code>', '').replace('</code>', '').replace('```python', '').replace('```', '')]

        # todo: explain PPL, LLM would output complete different format content, may causing PPL not correlated, such in for 22187233-48, DocNum=13, gpt-3.5
        # todo: also for 7961363-85, DocNum=10, llama
        # to explain the variation of performance, maybe can refer to the not formated output

        # todo: only for DocNum=5
        # if result['qs_id'] in ['18730044-52', '22187233-48'] and prompt_method == 'zero-shot' and 'gpt-3.5' in model: continue
        # todo: only for DocNum=7
        # if result['qs_id'] in ['8214932-94', '4172131-18', '4530069-65', '18730044-52'] and prompt_method == 'zero-shot' and 'gpt-3.5' in model: continue
        # todo: only for DocNum=10
        # if result['qs_id'] in ['674764-47', '15509617-37', '3283984-56', '2600191-99'] and prompt_method == 'zero-shot' and 'gpt-3.5' in model: continue
        # todo: only for DocNum=13
        # if result['qs_id'] in ['17731822-21', '15509617-37', '34945274-35', '4530069-65', '2212433-44', '18730044-52', '22187233-98', '22187233-48', '22187233-63'] and prompt_method == 'zero-shot' and 'gpt-3.5' in model: continue
        # todo: only for DocNum=16
        # if result['qs_id'] in ['19738169-65', '674764-33', '674764-47', '4530069-65', '22187233-98', '22187233-48', '22187233-75', '22187233-63'] and prompt_method == 'zero-shot' and 'gpt-3.5' in model: continue
        # todo: only for DocNum=20
        # if result['qs_id'] in ['14766194-35', '19738169-65', '674764-33', '674764-47', '17731822-21', '22187233-98', '22187233-48', '22187233-75', '22187233-63'] and prompt_method == 'zero-shot' and 'gpt-3.5' in model: continue

        # if result['qs_id'] in ['13628725-82', '31957364-53', '19779790-35', '23164058-67', '16296643-89']: continue

        if result['qs_id'] in ['8740353-74'] and prompt_method == 'zero-shot-CoT' and 'gpt-3.5' in model: continue
        if result['qs_id'] in ['2600191-98'] and prompt_method == 'few-shot' and 'gpt-3.5' in model: continue
        if result['qs_id'] in ['652291-62', '3283984-56', '20107570-40'] and prompt_method == 'self-refine' and 'gpt-3.5' in model: continue

        # todo: only for gpt-3.5 oracle
        # if result['qs_id'] in ['13628725-82', '31957364-53', '19779790-35', '23164058-67', '16296643-89', '7961363-7', '15819980-52', '10351772-57'] and prompt_method == 'zero-shot' and 'gpt-3.5' in model: continue

        # second: for some prompting methods, gpt would output extra reasoning, so parsing
        if prompt_method in ['CoT', 'Least-to-Most', 'self-refine', 'CoN'] and ('gpt-4o' in model or 'gpt-3.5' in model):
            # some cases that LLM cannot output runnable function
            assert result['response'].count('```') >= 2
            try:
                outputs = [result['response'].split('```python', 1)[1].split('```', 1)[0]]
            except:
                outputs = [result['response'].split('```', 1)[1].split('```', 1)[0]]

        # third for some other gpt-3.5, the llm would incorporate description into the python function, need parsing
        if 'gpt-3.5' in model:
            result['response'] = outputs[0]
            # if llm output both api + description + code
            lines = result['response'].split('\n')
            first_code_line_index = 0
            for idx, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('def ') or line.startswith('from '):
                    first_code_line_index = idx
                    break
            outputs = ['\n'.join(lines[first_code_line_index:])]
            lines = outputs[0].split('\n')
            for idx, line in enumerate(lines):
                if line.startswith('## Description:'):
                    del lines[idx+1]
                    break
            outputs = ['\n'.join(lines)]

        # finally: for codellama: split ```, or split [PYTHON]
        if 'codellama' in model:
            # todo: only for DocNum=3
            # if result['qs_id'] in ['23164058-67'] and prompt_method == 'zero-shot': continue
            # todo: only for DocNum=7
            # if result['qs_id'] in ['19738169-65', '13627865-82', '7961363-85'] and prompt_method == 'zero-shot': continue
            # todo: only for DocNum=10
            # if result['qs_id'] in ['11236006-9', '30693804-72', '574236-63', '7961363-85', '29903025-37', '40924332-83'] and prompt_method == 'zero-shot': continue
            # todo: only for DocNum=13
            # if result['qs_id'] in ['19738169-65', '30693804-72', '29836836-87', '29903025-37'] and prompt_method == 'zero-shot': continue
            # todo: only for DocNum=16
            # if result['qs_id'] in ['40055835-26', '19738169-65', '12496531-86', '4172131-18', '30693804-72', '574236-63', '7961363-7'] and prompt_method == 'zero-shot': continue
            # todo: only for DocNum=3
            if result['qs_id'] in ['13237941-14', '12496531-86', '13627865-82', '19758364-39', '574236-63', '7961363-7', '29836836-87'] and prompt_method == 'zero-shot': continue


            # some cases that LLM cannot output runnable function
            if result['qs_id'] in ['36139-62', '19779790-35', '19758364-39', '26153795-98', '29784889-2', '4172131-18', '574236-63', '4530069-65', '18367007-15'] and prompt_method == 'few-shot': continue
            if result['qs_id'] in ['23164058-67', '2600191-98'] and prompt_method == 'emotion': continue
            if result['qs_id'] in ['1883604-86', '4523551-62', '4172131-18', '15819980-52', '39816795-71'] and prompt_method == 'CoT': continue
            if result['qs_id'] in ['39816795-71', '23164058-67'] and prompt_method == 'zero-shot-CoT': continue
            if result['qs_id'] in ['39816795-71'] and prompt_method == 'Least-to-Most': continue
            if result['qs_id'] in ['19758364-39', '29370211-55', '19779790-35', '20573459-83', '11236006-9', '30693804-72', '8740353-74'] and prompt_method == 'self-refine': continue
            if result['qs_id'] in ['13237941-14', '652291-62', '1883604-86', '14766194-35', '19758364-39', '11236006-9', '7961363-86', '29836836-87', '20107570-40'] and prompt_method == 'CoN': continue

            result['response'] = result['response'].replace('```python', '```')
            try:
                assert result['response'].count('```') >= 2
                outputs = [result['response'].split('```', 1)[1].split('```', 1)[0]]
            except:
                assert '[/PYTHON]' in result['response'] and '[PYTHON]' in result['response']
                outputs = [result['response'].split('[PYTHON]', 1)[1].split('[/PYTHON]', 1)[0]]

        # finally check if we can extract the runnable function:
        assert outputs[0].strip() != ''
        assert check_python_code_ruff(unittests[result['qs_id']]['test'] + outputs[0]), print(result['qs_id'], outputs[0])

        _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))

    gene_results = list()
    for idx, result in enumerate(results):
        process_success = False
        for processed_result in _gene_results:
            if result['qs_id'] == processed_result['qs_id']:
                process_success = True
                gene_results.append(processed_result)
                break
        if process_success is False:
            gene_results.append(dict(qs_id=result['qs_id'], outputs=[result['response']]))
    return gene_results




def parsing_for_ds1000_new(qs_list, model, prompt_method, results):
    _gene_results = list()
    for idx, result in enumerate(results):
        assert qs_list[idx]['qs_id'] == result['qs_id']
        print(result['qs_id'])
        if result['qs_id'] == 'Pandas_50':
            print('ffffffffffff')
            print(result['response'])
        # first: default parsing for recall and docNum experiments:
        result['response'] = result['response'].replace('\nuse random_state=42', '').replace('Runnable code', '')
        outputs = [result['response'].replace('<code>', '').replace('</code>', '').replace('```python', '').replace('```', '')]

        # todo: only for gpt-4o oracle
        # if result['qs_id'] in ['Scipy_50'] and prompt_method == 'zero-shot' and 'gpt-4o' in model: continue
        # todo: only for gpt-4o single
        # if result['qs_id'] in ['Pandas_278'] and prompt_method == 'zero-shot' and 'gpt-4o' in model: continue
        # todo: only for gpt-4o DocNum=5
        if result['qs_id'] in ['Pytorch_20'] and prompt_method == 'zero-shot' and 'gpt-4o' in model: continue

        if result['qs_id'] in ['Numpy_140'] and prompt_method == 'few-shot' and 'gpt-4o' in model: continue
        if result['qs_id'] in ['Pytorch_64'] and prompt_method == 'Least-to-Most' and 'gpt-4o' in model: continue
        if result['qs_id'] in ['Pytorch_65'] and prompt_method == 'Plan-and-Solve' and 'gpt-4o' in model: continue
        if result['qs_id'] in ['Numpy_140', 'Scipy_50'] and prompt_method == 'self-refine' and 'gpt-4o' in model: continue

        if result['qs_id'] in ['Pandas_75', 'Pandas_97', 'Numpy_197', 'Scipy_14', 'Scipy_61'] and prompt_method == 'few-shot' and 'gpt-3.5' in model: continue
        if result['qs_id'] in ['Pandas_97', 'Pandas_72', 'Numpy_197', 'Sklearn_106'] and prompt_method == 'emotion' and 'gpt-3.5' in model: continue
        if result['qs_id'] in ['Pandas_97', 'Numpy_8', 'Scipy_14'] and prompt_method == 'CoT' and 'gpt-3.5' in model: continue
        if result['qs_id'] in ['Pandas_97', 'Numpy_197'] and prompt_method == 'zero-shot-CoT' and 'gpt-3.5' in model: continue
        if result['qs_id'] in ['Pandas_97', 'Numpy_8', 'Tensorflow_1', 'Scipy_14'] and prompt_method == 'Least-to-Most' and 'gpt-3.5' in model: continue
        if result['qs_id'] in ['Pandas_290', 'Numpy_138', 'Numpy_8', 'Numpy_133'] and prompt_method == 'self-refine' and 'gpt-3.5' in model: continue
        if result['qs_id'] in ['Scipy_23'] and prompt_method == 'self-refine' and 'gpt-3.5' in model: continue
        if result['qs_id'] in ['Numpy_22', 'Numpy_8'] and prompt_method == 'CoN' and 'gpt-3.5' in model: continue
        if prompt_method == 'Plan-and-Solve' and 'gpt-3.5' in model and result['qs_id'] in ['Pandas_258', 'Numpy_133', 'Scipy_102', 'Pandas_97', 'Pandas_278', 'Numpy_98', 'Scipy_14', 'Scipy_15']: continue
        if prompt_method in ['Plan-and-Solve', 'zero-shot-CoT'] and 'gpt-3.5' in model:
            try:
                result['response'] = '```' + result['response'].split('## Complete Code:', 1)[1]
            except:
                try:
                    result['response'] = '```' + result['response'].split('## Incomplete Code:', 1)[1]
                except:
                    ...



        # todo: ds1000 主要需要匹配删除已有的prompt code, llm在recall很小的时候可能反而生成的code和原有prompt code更一致
        # second: for some prompting methods, gpt would output extra reasoning, so parsing
        if prompt_method in ['few-shot', 'CoT', 'Least-to-Most', 'self-refine', 'CoN', 'Plan-and-Solve', 'zero-shot-CoT'] and ('gpt-4o' in model or 'gpt-3.5' in model):
            result['response'] = result['response'].replace('<code>', '```python').replace('</code>', '```')
            result['response'] = result['response'].replace('```python', '```')
            assert result['response'].count('```') >= 2
            outputs = [result['response'].split('```', 1)[1].split('```', 1)[0]]

        # finally: for codellama: split ```, or split [PYTHON]
        elif 'codellama' in model:
            # todo: only for DocNum=5
            # if result['qs_id'] in ['Pandas_252', 'Sklearn_59'] and prompt_method == 'initial_output': continue
            if result['qs_id'] in ['Pandas_252', 'Sklearn_59'] and prompt_method == 'zero-shot': continue


            if result['qs_id'] in ['Pandas_248', 'Pandas_244', 'Pandas_50', 'Pandas_37', 'Pandas_278', 'Numpy_179', 'Scipy_14', 'Sklearn_33', 'Sklearn_12'] and prompt_method == 'CoN': continue
            if result['qs_id'] in ['Pandas_155'] and prompt_method == 'few-shot': continue
            if result['qs_id'] in ['Sklearn_25'] and prompt_method == 'emotion': continue
            if result['qs_id'] in ['Numpy_31', 'Numpy_218', 'Scipy_12', 'Sklearn_28'] and prompt_method == 'CoT': continue
            if result['qs_id'] in ['Scipy_61', 'Sklearn_9'] and prompt_method == 'zero-shot-CoT': continue
            if result['qs_id'] in ['Pandas_55', 'Sklearn_8'] and prompt_method == 'Least-to-Most': continue
            if result['qs_id'] in ['Pandas_272', 'Scipy_102', 'Sklearn_106', 'Sklearn_33'] and prompt_method == 'Plan-and-Solve': continue
            if result['qs_id'] in ['Pandas_280', 'Numpy_20', 'Scipy_50', 'Scipy_14', 'Sklearn_106', 'Sklearn_12'] and prompt_method == 'self-refine': continue


            if prompt_method in ['zero-shot-CoT', 'CoT', 'Least-to-Most', 'self-refine'] and '[PYTHON]' in result['response'] and not '[/PYTHON]' in result['response']:
                result['response'] = result['response'].replace('```', '[/PYTHON]')

            result['response'] = result['response'].replace('```python', '```')
            try:
                assert result['response'].count('```') >= 2, print(result['response'])
                outputs = [result['response'].split('```', 1)[1].split('```', 1)[0]]
            except:
                assert '[/PYTHON]' in result['response'] and '[PYTHON]' in result['response']
                # if not result['qs_id'] == 'Scipy_50':  # todo: doc num k = 1
                outputs = [result['response'].split('[PYTHON]', 1)[1].split('[/PYTHON]', 1)[0]]

        # finally check if we can extract the runnable function:
        assert outputs[0].strip() != ''
        assert check_python_code_ruff(outputs[0]), print(result['qs_id'], outputs[0])

        # finally matching and replace existing code
        if not prompt_method == 'initial_output':
            outputs = process_ds1000_outputs(pid=result['qs_id'], outputs=outputs, existing_code=qs_list[idx]['question'].split('\nA:')[1])

        _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))

    gene_results = list()
    for idx, result in enumerate(results):
        process_success = False
        for processed_result in _gene_results:
            if result['qs_id'] == processed_result['qs_id']:
                process_success = True
                gene_results.append(processed_result)
                break
        if process_success is False:
            gene_results.append(dict(qs_id=result['qs_id'], outputs=[result['response']]))
    return gene_results


def parsing_for_pne_new(qs_list, model, prompt_method, results):
    _gene_results = []
    for idx, result in enumerate(results):
        assert qs_list[idx]['qs_id'] == result['qs_id']
        print(result['qs_id'])

        # Before All skip identified failed program
        if prompt_method == 'few-shot' and 'gpt-4o' in model and result['qs_id'] in ['NumpyEval/4']: continue
        if prompt_method == 'emotion' and 'gpt-4o' in model and result['qs_id'] in ['PandasEval/38']: continue
        if prompt_method == 'zero-shot-CoT' and 'gpt-4o' in model and result['qs_id'] in ['PandasEval/38']: continue
        if prompt_method == 'self-refine' and 'gpt-4o' in model and result['qs_id'] in ['PandasEval/14', 'PandasEval/82']: continue
        if prompt_method == 'CoN' and 'gpt-3.5' in model and result['qs_id'] in ['PandasEval/54', 'PandasEval/90', 'NumpyEval/2', 'NumpyEval/42']: continue
        if prompt_method == 'self-refine' and 'gpt-3.5' in model and result['qs_id'] in ['NumpyEval/2']: continue
        if prompt_method == 'Plan-and-Solve' and 'gpt-3.5' in model and result['qs_id'] in ['NumpyEval/2', 'NumpyEval/87']: continue
        if prompt_method == 'Least-to-Most' and 'gpt-3.5' in model and result['qs_id'] in ['NumpyEval/9', 'NumpyEval/87']: continue
        if prompt_method == 'zero-shot-CoT' and 'gpt-3.5' in model and result['qs_id'] in ['NumpyEval/2', 'NumpyEval/4']: continue
        if prompt_method == 'CoT' and 'gpt-3.5' in model and result['qs_id'] in ['NumpyEval/4', 'NumpyEval/56']: continue
        if prompt_method == 'emotion' and 'gpt-3.5' in model and result['qs_id'] in ['NumpyEval/2', 'NumpyEval/4', 'NumpyEval/39']: continue

        # todo: only for docnum=5, codellama, self-refine usage
        # if prompt_method == 'initial_output' and 'codellama' in model and result['qs_id'] in ['PandasEval/32', 'NumpyEval/34']: continue

        # todo: only for docnum=5 usage
        # if prompt_method == 'zero-shot' and 'gpt-3.5' in model and result['qs_id'] in ['NumpyEval/2']: continue
        if prompt_method == 'zero-shot' and 'gpt-4o' in model and result['qs_id'] in ['PandasEval/38']: continue


        if prompt_method in ['Plan-and-Solve', 'zero-shot-CoT'] and 'gpt-3.5' in model:
            if result['response'].replace('\n','').startswith('```<code>'): result['response'] = result['response'].replace('<code>', '').replace('</code>', '')

        # first: default parsing for recall and docNum experiments:
        result['response'] = result['response'].replace('```code', '```')
        outputs = [result['response'].replace('<code>', '').replace('</code>', '').replace('```python', '').replace('```', '')]


        # second: for some prompting methods, gpt would output extra reasoning, so parsing
        if prompt_method in ['CoT', 'Least-to-Most', 'self-refine', 'CoN', 'zero-shot-CoT', 'Plan-and-Solve'] and ('gpt-4o' in model or 'gpt-3.5' in model):
            result['response'] = result['response'].replace('<code>', '```python').replace('</code>', '```')
            result['response'] = result['response'].replace('```python', '```')
            assert result['response'].count('```') >= 2
            outputs = [result['response'].split('```', 1)[1].split('```', 1)[0]]

        # finally: for codellama: split ```, or split [PYTHON]
        elif 'codellama' in model:
            if prompt_method == 'self-refine' and 'Refined solution' in result['response']:
                result['response'] = result['response'].split('Refined solution')[1]
            if prompt_method in ['few-shot', 'CoT', 'Least-to-Most', 'self-refine']:
                if '## API Documents:' in result['response']:
                    result['response'] = result['response'].split('## API Documents:')[0]
                    if result['response'].count('```') < 2: result['response'] = result['response'] + '```'
                if result['response'].count('```') == 1 and '[PYTHON]' in result['response']: result['response'] = result['response'].replace('[PYTHON]', '```')
                # if '```[/PYTHON]' in result['response'].replace('\n', ''): result['response'] = result['response'].replace('[PYTHON]', '```').replace('[/PYTHON]', '')

            if prompt_method == 'few-shot' and result['qs_id'] in ['NumpyEval/39']: continue
            if prompt_method == 'emotion' and result['qs_id'] in ['PandasEval/32']: continue
            if prompt_method == 'CoT' and result['qs_id'] in ['NumpyEval/60']: continue
            if prompt_method == 'zero-shot-CoT' and result['qs_id'] in ['NumpyEval/18', 'NumpyEval/100']: continue
            if prompt_method == 'Least-to-Most' and result['qs_id'] in ['NumpyEval/60']: continue
            if prompt_method == 'Plan-and-Solve' and result['qs_id'] in ['NumpyEval/89']: continue
            if prompt_method == 'self-refine' and result['qs_id'] in ['NumpyEval/96']: continue
            # if prompt_method == 'CoN' and result['qs_id'] in ['PandasEval/3'']: continue

            # todo: only for doc num=5
            if prompt_method == 'zero-shot' and result['qs_id'] in ['NumpyEval/34']: continue


            # first remove space at the front of generation, only happen in PNE
            if result['response'].startswith(' '): result['response'] = result['response'][1:]
            assert not result['response'].startswith(' ')
            # remove last generation sign " package"
            result['response'] = result['response'].replace(' package', '')
            # then try to split and get only the code
            try:
                assert result['response'].count('```') >= 2
                result['response'] = result['response'].replace('```python', '```')
                outputs = [result['response'].split('```', 1)[1].split('```', 1)[0]]
            except:
                # remove content after response
                # if 'return' in result['response'] or 'print' in result['response']:
                #     lines = result['response'].split('\n')
                #     lines = [line for line in lines if not line.strip().startswith('#')]
                #     outputs = []
                #     for line in lines:
                #         outputs.append(line)
                #         if line.strip().startswith('return') or line.strip().startswith('print'): break
                #     outputs = ['\n'.join(outputs)]
                #     print(outputs)
                assert '[/PYTHON]' in result['response'] and '[PYTHON]' in result['response']
                # if not result['qs_id'] == 'Scipy_50':  # todo: doc num k = 1
                outputs = [result['response'].split('[PYTHON]', 1)[1].split('[/PYTHON]', 1)[0]]

        # finally check if we can extract the runnable function:
        outputs = ['import pandas as pd\nimport numpy as np\n' + outputs[0]]
        assert outputs[0].strip() != ''
        assert check_python_code_ruff(outputs[0]), print(result['qs_id'], outputs)

        _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))

    gene_results = list()
    for idx, result in enumerate(results):
        process_success = False
        for processed_result in _gene_results:
            if result['qs_id'] == processed_result['qs_id']:
                process_success = True
                gene_results.append(processed_result)
                break
        if process_success is False:
            gene_results.append(dict(qs_id=result['qs_id'], outputs=[result['response']]))
    return gene_results


model_names_for_path = {"gpt-4o-mini": "gpt-4o-mini",
                                     "gpt-3.5-turbo-0125": "gpt-3-5-turbo",
                                     "codellama/CodeLlama-13b-Instruct-hf": "codellama-13b",
                                     "meta-llama/Llama-2-13b-chat-hf": 'llama2-13b'}


def parsing_for_qa_new(qs_list, model, prompt_method, results):
    _gene_results = []
    for idx, result in enumerate(results):
        assert qs_list[idx]['qs_id'] == result['qs_id']

        # outputs = [result['response'].replace('<code>', '').replace('</code>', '').replace('```python', '').replace('```', '')]
        pred = result['response']
        try:
            assert '<answer>' in pred and '</answer>' in pred
            pred = pred.split('<answer>')[1].split('</answer>')[0]
        except:
            try:
                assert pred.count('```') == 2
                pred = pred.split('```')[1].split('```')[0]
            except:
                ...

        _gene_results.append(dict(qs_id=result['qs_id'], outputs=[pred]))
    return _gene_results

model_names_for_path = {"gpt-4o-mini": "gpt-4o-mini",
                                     "gpt-3.5-turbo-0125": "gpt-3-5-turbo",
                                     "codellama/CodeLlama-13b-Instruct-hf": "codellama-13b",
                                     "meta-llama/Llama-2-13b-chat-hf": 'llama2-13b'}

def pred_eval_new(model, dataset, prompt_method, result_path):
    eval_save_file = result_path.replace('.json', '_eval.json')
    results = json.load(open(result_path, 'r'))
    if dataset in ['CoNaLa', 'DS1000', 'pandas_numpy_eval']: k = 5
    else: k = 10
    # results_before = json.load(open(f'../data/{self.dataset}/new_results/DocNum/{k}_{self.model_names_for_path[self.model_config.model]}.json', 'r'))
    results_before = json.load(open(f'../data/{dataset}/new_results/Prompt/few-shot_{model_names_for_path[model]}.json', 'r'))

    if dataset == 'conala':
        loader = ConalaLoader()
        qs_list = loader.load_qs_list()
        _gene_results = parsing_for_conala_new(qs_list=qs_list, model=model, prompt_method=prompt_method, results=results)
        scores, eval_records = loader.eval_passk(_gene_results, top_k=[1])
        syntax_error_count = 0
        for qid in eval_records:
            if eval_records[qid]['syntax_error']:
                syntax_error_count += 1
        print('number of syntax errors: {}'.format(syntax_error_count))
    elif dataset == 'DS1000':
        loader = DS1000Loader()
        qs_list = loader.load_qs_list()
        _gene_results = parsing_for_ds1000_new(qs_list=qs_list, model=model, prompt_method=prompt_method, results=results)
        scores, eval_records = loader.eval_passk(_gene_results, k_list=[1])
        syntax_error_count = 0
        for idx, qid in enumerate(eval_records):
            if eval_records[qid]['syntax_error']:
                syntax_error_count += 1
                # print(_gene_results[idx]['outputs'][0])
        print('number of syntax errors: {}'.format(syntax_error_count))
    elif dataset == 'pandas_numpy_eval':
        loader = PandasNumpyEvalLoader()
        qs_list = loader.load_qs_list()
        _gene_results = parsing_for_pne_new(qs_list=qs_list, model=model, prompt_method=prompt_method, results=results)
        scores, eval_records = loader.eval_passk(_gene_results, k_list=[1])
        syntax_error_count = 0
        for qid in eval_records:
            if eval_records[qid]['syntax_error']:
                syntax_error_count += 1
        print('number of syntax errors: {}'.format(syntax_error_count))
    elif dataset == 'NQ' or dataset == 'TriviaQA':
        loader = NQTriviaQAUtils(dataset)
        oracle_list = loader.load_oracle_list()
        preds, answers_list = [], []
        for idx, (result, oracle) in enumerate(zip(results, oracle_list)):
            assert str(result['qs_id']) == str(oracle['qs_id'])
            if 'response' in result.keys():
                pred = process_gene_results(result['qs_id'], dataset, [result['response']], prompt_type=prompt_method, outputs_before=results_before[idx]['outputs'])[0]   # no k trial for QA datasets
            else:
                assert len(result['outputs']) == 1
                pred = process_gene_results(result['qs_id'], dataset, result['outputs'], prompt_type=prompt_method, outputs_before=results_before[idx]['outputs'])[0]
            preds.append(pred)
            answers_list.append(oracle['answers'])
        scores, _eval_records = loader.pred_eval(preds=preds, answers_list=answers_list)
        eval_records = dict()
        for idx, oracle in enumerate(oracle_list):
            eval_records[oracle['qs_id']] = _eval_records[idx]
    elif dataset == 'hotpotQA':
        loader = HotpotQAUtils()
        oracle_list = loader.load_oracle_list()
        pred_list = []
        for idx, result in enumerate(results):
            if 'response' in result.keys():
                output = process_gene_results(result['qs_id'], dataset, [result['response']], prompt_type=prompt_method, outputs_before=results_before[idx]['outputs'])[0]   # Todo: now only 1 inference
            else:
                assert len(result['outputs']) == 1
                output = process_gene_results(result['qs_id'], dataset, result['outputs'], prompt_type=prompt_method, outputs_before=results_before[idx]['outputs'])[0]
            pred_list.append(dict(qs_id=result['qs_id'], output=output))    # format for eval_pred()
        scores, eval_records = loader.eval_pred(pred_list=pred_list, oracle_list=oracle_list)
    else:
        raise ValueError('Not supported dataset {}'.format(dataset))
    scores = {key: round(value, 3) for key, value in scores.items() if value is not None}
    print(scores)
    with open(eval_save_file, 'w') as f:
        json.dump(dict(scores=scores, eval_records=eval_records), f, indent=2)




def pred_eval(args, if_eval_retrieval=False, if_calc_perplexity=True, if_code_analysis=True, if_save=True):
    eval_save_file = args.result_save_file.replace('.json', '_eval.json')
    # if os.path.exists(eval_save_file):
    #     print('eval file exists already, {}'.format(eval_save_file))
    #     eval_results = json.load(open(eval_save_file, 'r'))
    #     print(eval_results['scores'])
    #     return

    gene_results = json.load(open(args.result_save_file, 'r'))
    if args.prompt_type == 'self-refine': gene_results_before = json.load(open(args.result_save_file.replace('self-refine', '3shot'), 'r'))
    # if args.n == 10:
    #     k_list = [1,3,5,10]
    # elif args.n == 1:
    #     k_list = [1]
    # else:
    #     raise ValueError('args.n must be 1 or 10')
    k_list = [1]

    output_records = dict()
    retrieval_records = dict()



    if args.dataset == 'conala':
        loader = ConalaLoader()
        _gene_results = list()
        for idx, result in enumerate(gene_results):
            if args.prompt_type == 'self-refine': outputs = process_gene_results(args, result['outputs'], outputs_before=gene_results_before[idx]['outputs'])
            elif args.prompt_type == 'self-consistency': outputs = [process_outputs_for_self_consistency(process_gene_results(args, result['outputs']))]
            else: outputs = process_gene_results(args, result['outputs'])
            # outputs = [result['oracle_output']]
            _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
            output_records[result['qs_id']] = outputs
            retrieval_records[result['qs_id']] = result['ret_docs']
        scores, eval_records = loader.eval_passk(_gene_results, top_k=k_list)

    elif args.dataset == 'DS1000':
        # gene_results = json.load(open(DS1000Loader().oracle_doc_file, 'r'))
        loader = DS1000Loader()
        qs_list = loader.load_qs_list()
        _gene_results = list()
        for idx, result in enumerate(gene_results):
            assert qs_list[idx]['qs_id'] == result['qs_id']
            if args.prompt_type == 'self-refine': outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'].split('\nA:')[1], outputs_before=gene_results_before[idx]['outputs'])
            elif args.prompt_type == 'self-consistency': outputs = [process_outputs_for_self_consistency(process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'].split('\nA:')[1]))]
            else: outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'].split('\nA:')[1])
            _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
            output_records[result['qs_id']] = outputs
            retrieval_records[result['qs_id']] = result['ret_docs']
        scores, eval_records = loader.eval_passk(_gene_results, k_list=k_list)
        # scores, eval_records = dict(), dict()

    elif args.dataset == 'pandas_numpy_eval':
        loader = PandasNumpyEvalLoader()
        qs_list = loader.load_qs_list()
        _gene_results = []
        for idx, result in enumerate(gene_results):
            assert qs_list[idx]['qs_id'] == result['qs_id']
            if args.prompt_type == 'self-refine': outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'], outputs_before=gene_results_before[idx]['outputs'])
            elif args.prompt_type == 'self-consistency': outputs = [process_outputs_for_self_consistency(process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question']))]
            else: outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'])
            _gene_results.append(dict(qs_id=result['qs_id'], outputs=outputs))
            output_records[result['qs_id']] = outputs
            retrieval_records[result['qs_id']] = result['ret_docs']
        scores, eval_records = loader.eval_passk(_gene_results, k_list=k_list)
        # scores, eval_records = dict(), dict()

    elif args.dataset == 'NQ' or args.dataset == 'TriviaQA':
        loader = NQTriviaQAUtils(args.dataset)
        oracle_list = loader.load_oracle_list()
        preds, answers_list = [], []
        for idx, (result, oracle) in enumerate(zip(gene_results, oracle_list)):
            assert str(result['qs_id']) == str(oracle['qs_id'])
            if args.prompt_type == 'self-refine': pred = process_gene_results(args, result['outputs'], outputs_before=gene_results_before[idx]['outputs'])[0]
            elif args.prompt_type == 'self-consistency': pred = process_outputs_for_self_consistency(process_gene_results(args, result['outputs']))
            else: pred = process_gene_results(args, result['outputs'])[0]  # Todo: now only 1 inference
            preds.append(pred)
            answers_list.append(oracle['answers'])
            output_records[result['qs_id']] = [pred]
            retrieval_records[result['qs_id']] = result['ret_docs']
        scores, _eval_records = loader.pred_eval(preds=preds, answers_list=answers_list)
        eval_records = dict()
        for idx, oracle in enumerate(oracle_list):
            eval_records[oracle['qs_id']] = _eval_records[idx]

    elif args.dataset == 'hotpotQA':
        loader = HotpotQAUtils()
        oracle_list = loader.load_oracle_list()
        pred_list = []
        for idx, result in enumerate(gene_results):
            if args.prompt_type == 'self-refine': output = process_gene_results(args, result['outputs'], outputs_before=gene_results_before[idx]['outputs'])[0]
            elif args.prompt_type == 'self-consistency': output = process_outputs_for_self_consistency(process_gene_results(args, result['outputs']))
            else: output = process_gene_results(args, result['outputs'])[0]   # Todo: now only 1 inference
            pred_list.append(dict(qs_id=result['qs_id'], output=output))
            output_records[result['qs_id']] = [output]
            retrieval_records[result['qs_id']] = result['ret_docs']
        scores, eval_records = loader.eval_pred(pred_list=pred_list, oracle_list=oracle_list)

    else:
        raise ValueError('Not supported dataset {}'.format(args.dataset))

    ret_doc_keys_list, prompts, pl_list = [], [], []
    with open(args.prompt_save_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if len(data['ret_doc_keys']) != 0: ret_doc_keys_list.append(data['ret_doc_keys'])
            prompts.append(data['prompt'])
            pl_list.append(data['prompt_length'])

    ret_doc_key_flags_list = None
    if len(ret_doc_keys_list) != 0 and if_eval_retrieval is True:
        oracle_list = loader.load_oracle_list()
        ret_doc_key_flags_list, avg_ret_recall, avg_oracle_percent, avg_oracle_rank = ret_eval_by_doc_keys(dataset=args.dataset, oracle_list=oracle_list, ret_doc_keys_list=ret_doc_keys_list)
        # print('ret recall: ', avg_ret_recall)
        # print('avg oracle doc percentage: ', avg_oracle_percent)
        # print('avg oracle doc rank: ', avg_oracle_rank + 1)  # rank start from 1
        # print('avg prompt length: ', sum(pl_list) / len(pl_list))
        scores['ret_recall'] = avg_ret_recall
        scores['oracle_percent'] = avg_oracle_percent
        scores['oracle_rank'] = avg_oracle_rank
        print('ishjdsfeuhisjocjseghijso')

    # avg prompt length
    scores['prompt_length'] = sum(pl_list) / len(pl_list)


    # calc perplexity
    if if_calc_perplexity is True:
        perplexity = 0
        perplexity_list = []
        batch_idx = 0; valid_outputs = []; logprobs_list = []
        for result_idx, result in enumerate(gene_results):
            logprobs = result['logprobs'][0]  # todo: only for n=1
            # llama would output extra content, remove them when calculating perplexity
            if 'llama' in args.model and args.analysis_type == 'prompt_method':
                batch_idx += 1
                valid_outputs.append(result['outputs'][0].split('Potential documents')[0].replace('\n\n\n', ''))
                logprobs_list.append(logprobs[0])   # for llama
                if batch_idx == 100 or result_idx == len(gene_results) - 1:
                    valid_outputs_length = get_docs_tokens(valid_outputs, args.model)
                    logprobs_list = [logprobs[:length] for logprobs, length in zip(logprobs_list, valid_outputs_length)]
                    for logprobs in logprobs_list:
                        perplexity += np.exp(-sum(logprobs) / len(logprobs))
                        perplexity_list.append(np.exp(-sum(logprobs) / len(logprobs)))
                    batch_idx = 0; valid_outputs = []; logprobs_list = []
            else:
                if 'llama' in args.model:
                    logprobs = logprobs[0]  # for llama
                try:
                    perplexity += np.exp(-sum(logprobs) / len(logprobs))
                    perplexity_list.append(np.exp(-sum(logprobs) / len(logprobs)))
                except: print(logprobs)
        # scores['perplexity'] = perplexity / len(gene_results)
        # oracle_list = loader.load_oracle_list()
        # for idx, oracle in enumerate(oracle_list):
        #     eval_records[oracle['qs_id']]['perplexity'] = perplexity_list[idx]
        print(perplexity_list)


    # extra analyze for code
    if if_code_analysis is True and args.dataset in ['conala', 'DS1000', 'pandas_numpy_eval']:
        eval_datas = dict(eval_records=eval_records, output_records=output_records, retrieval_records=retrieval_records, ret_eval_records=ret_doc_key_flags_list if len(ret_doc_keys_list) != 0 else [])
        retrieval_consistency, syntax_error, semantic_error = analyze_results_for_code(args.dataset, eval_datas)
        scores['retrieval_consistency'] = retrieval_consistency
        scores['syntax_error_percent'] = syntax_error
        scores['semantic_error_percent'] = semantic_error

    scores = {key: round(value, 3) for key, value in scores.items() if value is not None}
    print(scores)
    if if_save:
        with open(eval_save_file, 'w') as f:
            json.dump(dict(scores=scores, eval_records=eval_records, output_records=output_records, retrieval_records=retrieval_records, ret_eval_records=ret_doc_key_flags_list), f, indent=2)

    return scores


"""

if __name__ == '__main__':
    in_program_call = None
    ret_accs = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for ret_acc in ret_accs:
        in_program_call = f'--model codellama-13b-instruct --dataset DS1000 --retriever openai-embedding --analysis_type prompt_method --prompt_type 3shot --n 1'
        # in_program_call = '--model codellama-13b-instruct --dataset conala --retriever openai-embedding --n 1 --analysis_type retrieval_doc_selection --doc_selection_type top_5'
        # args = generate_config(in_program_call)

        # scores = pred_eval(args, if_eval_retrieval=False, if_code_analysis=True, if_calc_perplexity=False, if_save=True)

    # if args.dataset == 'DS1000':
    #     """
    #     test process outputs for DS1000
    #     """
    #     cannot_answer_count = 0
    #     ds1000 = DS1000Dataset(source_dir='../data/DS1000/ds1000_data', libs='all', mode='Insertion')
    #     gene_results = json.load(open(args.result_save_file, 'r'))
    #     if args.prompt_type == 'self-refine': gene_results_before = json.load(open(args.result_save_file.replace('self-refine', '3shot'), 'r'))
    #     loader = DS1000Loader()
    #     oracle_list = loader.load_oracle_list()
    #     qs_list = loader.load_qs_list()
    #     for idx, result in enumerate(gene_results):
    #         qs_id = result['qs_id']
    #         [lib, problema_id] = qs_id.split('_')
    #         data = ds1000[lib][int(problema_id)]
    #         print(f'\n<processed code {idx}>]')
    #         print([result['outputs'][0]])
    #         if '\n\n\n\n\n\n\n\n' in result['outputs'][0]: cannot_answer_count += 1
    #         # print([result['outputs'][0]])
    #         # print(qs_list[idx]['question'].split('\nA:')[1])
    #         if args.prompt_type == 'self-refine':
    #             outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'].split('\nA:')[1], outputs_before=gene_results_before[idx]['outputs'])
    #         else:
    #             outputs = process_gene_results(args, result['outputs'], code_prompt=qs_list[idx]['question'].split('\nA:')[1])
    #         if args.prompt_type == 'self-consistency':
    #             print(outputs)
    #             most_output = process_outputs_for_self_consistency(outputs)
    #             print(most_output)
    #         else:
    #             print([outputs[0]])
    #     print(cannot_answer_count)
    #
    # elif args.dataset == 'pandas_numpy_eval':
    #     """
    #     test process outputs for pandas_numpy_eval
    #     """
    #     dataset = json.load(open('../data/pandas_numpy_eval/data/pandas_numpy_eval.json', 'r'))
    #     gene_results = json.load(open(args.result_save_file, 'r'))
    #     if args.prompt_type == 'self-refine': gene_results_before = json.load(open(args.result_save_file.replace('self-refine', '3shot'), 'r'))
    #     for idx, result in enumerate(gene_results):
    #         print(f'\n<processed code {idx}>]')
    #         print([result['outputs'][0]])
    #         # print([result['outputs'][0]])
    #         for data in dataset:
    #             if data['task_id'] == result['qs_id']:
    #                 code_prompt = data['prompt']
    #         if args.prompt_type == 'self-refine':
    #             outputs = process_gene_results(args, result['outputs'], code_prompt, gene_results_before[idx]['outputs'])
    #         else:
    #             outputs = process_gene_results(args, result['outputs'], code_prompt)
    #         if args.prompt_type == 'self-consistency':
    #             print(outputs)
    #             most_output = process_outputs_for_self_consistency(outputs)
    #             print(most_output)
    #         else:
    #             print([outputs[0]])
    #
    # elif args.dataset == 'conala':
    #     """
    #     test for conala
    #     """
    #     gene_results = json.load(open(args.result_save_file, 'r'))
    #     if args.prompt_type == 'self-refine': gene_results_before = json.load(open(args.result_save_file.replace('self-refine', '3shot'), 'r'))
    #     for idx, result in enumerate(gene_results):
    #         print(f'\n<processed code {idx}>]')
    #         print([result['outputs'][0]])
    #         if args.prompt_type == 'self-refine':
    #             outputs = process_gene_results(args, result['outputs'], outputs_before=gene_results_before[idx]['outputs'])
    #         else:
    #             outputs = process_gene_results(args, result['outputs'])
    #         if args.prompt_type == 'self-consistency':
    #             print(outputs)
    #             most_output = process_outputs_for_self_consistency(outputs)
    #             print(most_output)
    #         else:
    #             print([outputs[0]])
    #
    # else:
    #     """
    #     test for NQ, TriviaQA, hotpotQA
    #     """
    #     cannot_answer_count = 0
    #     gene_results = json.load(open(args.result_save_file, 'r'))
    #     if args.prompt_type == 'self-refine': gene_results_before = json.load(open(args.result_save_file.replace('self-refine', '3shot'), 'r'))
    #     for idx, result in enumerate(gene_results):
    #         print(f'\n<processed answer {idx}>]')
    #         print([result['outputs']])
    #         if "I'm sorry" in result['outputs'][0]: cannot_answer_count += 1
    #         if args.prompt_type == 'self-refine':
    #             outputs = process_gene_results(args, result['outputs'], outputs_before=gene_results_before[idx]['outputs'])
    #         else:
    #             outputs = process_gene_results(args, result['outputs'])
    #         if args.prompt_type == 'self-consistency':
    #             print(outputs)
    #             most_output = process_outputs_for_self_consistency(outputs)
    #             print(most_output)
    #         else:
    #             print([outputs[0]])
    #     print(cannot_answer_count)
    #
    # # todo: for self-consistency
