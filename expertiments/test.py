import tempfile
import os
import subprocess


def check_python_code_ruff(code_string):
    """
    Check Python code string using Ruff for errors only (ignoring warnings)

    Args:
        code_string (str): Python code to check

    Returns:
        bool: True if no errors found, False if errors detected
    """
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

            # Return True if no errors (exit code 0), False if errors found
            # print(result)
            return result.returncode == 0

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


if __name__ == '__main__':
    code = "code\nimport pandas as pd\nimport numpy as np\n\ndef replacing_blank_with_nan(df):\n    # replace field that's entirely space (or empty) with NaN using regex\n    df.replace(r'^\\s*$', np.nan, regex=True, inplace=True)\n    return df\n"
    result = check_python_code_ruff(code)
    print(result)
