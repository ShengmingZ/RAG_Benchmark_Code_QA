tldr_original_3shots_prompt = '''Potential document 0: fatlabel_3: fatlabel will display or change the volume label or volume ID on the MS- DOS filesystem located on DEVICE.  By default it works in label mode.  It can be switched to volume ID mode with the option -i or --volume-id.
# get the label of a fat32 partition
fatlabel {{/dev/sda1}}

#END

Potential document 0: w_3: w displays information about the users currently on the machine, and their processes.  The header shows, in this order, the current time, how long the system has been running, how many users are currently logged on, and the system load averages for the past 1, 5, and 15 minutes.
Potential document 1: w_9: -s, --short Use the short format.  Don't print the login time, JCPU or PCPU times.
# display information without including the login, jcpu and pcpu columns
w --short

#END

Potential document 0: csvsort_2: Sort CSV files. Like the Unix “sort” command, but for tabular data:
Potential document 1: csvsort_3: usage: csvsort [-h] [-d DELIMITER] [-t] [-q QUOTECHAR] [-u {0,1,2,3}] [-b] [-p ESCAPECHAR] [-z FIELD_SIZE_LIMIT] [-e ENCODING] [-L LOCALE] [-S] [--blanks] [--date-format DATE_FORMAT] [--datetime-format DATETIME_FORMAT] [-H] [-K SKIP_LINES] [-v] [-l] [--zero] [-V] [-n] [-c COLUMNS] [-r] [-y SNIFF_LIMIT] [-I
Potential document 2: csvsort_6: optional arguments: -h, --help            show this help message and exit -n, --names           Display column names and indices from the input CSV and exit. -c COLUMNS, --columns COLUMNS A comma separated list of column indices, names or ranges to sort by, e.g. "1,id,3-5". Defaults to all columns. -r, --reverse         Sort in descending order. -y SNIFF_LIMIT, --snifflimit SNIFF_LIMIT Limit CSV dialect sniffing to the specified number of bytes. Specify "
Potential document 3: csvsort_10: csvsort -c 9 examples/realdata/FY09_EDU_Recipients_by_State.csv
Potential document 4: csvsort_12: csvcut -c 1,9 examples/realdata/FY09_EDU_Recipients_by_State.csv | csvsort -r -c 2 | head -n 5
# sort a csv file by column 9
csvsort -c {{9}} {{data.csv}}

#END'''

tldr_3shots_prompt_with_instruction = '''Given the description, and some potential documents that might help, generate corresponding command,
Only generate the command, and all variables in the command should be in the format of {{VAR}} and do not include "shell" or "bash". 
And pay attention that potential documents might not be helpful when generating the command
Potential document 0: fatlabel_3: fatlabel will display or change the volume label or volume ID on the MS- DOS filesystem located on DEVICE.  By default it works in label mode.  It can be switched to volume ID mode with the option -i or --volume-id.
# get the label of a fat32 partition
fatlabel {{/dev/sda1}}

#END

Potential document 0: w_3: w displays information about the users currently on the machine, and their processes.  The header shows, in this order, the current time, how long the system has been running, how many users are currently logged on, and the system load averages for the past 1, 5, and 15 minutes.
Potential document 1: w_9: -s, --short Use the short format.  Don't print the login time, JCPU or PCPU times.
# display information without including the login, jcpu and pcpu columns
w --short

#END

Potential document 0: csvsort_2: Sort CSV files. Like the Unix “sort” command, but for tabular data:
Potential document 1: csvsort_3: usage: csvsort [-h] [-d DELIMITER] [-t] [-q QUOTECHAR] [-u {0,1,2,3}] [-b] [-p ESCAPECHAR] [-z FIELD_SIZE_LIMIT] [-e ENCODING] [-L LOCALE] [-S] [--blanks] [--date-format DATE_FORMAT] [--datetime-format DATETIME_FORMAT] [-H] [-K SKIP_LINES] [-v] [-l] [--zero] [-V] [-n] [-c COLUMNS] [-r] [-y SNIFF_LIMIT] [-I
Potential document 2: csvsort_6: optional arguments: -h, --help            show this help message and exit -n, --names           Display column names and indices from the input CSV and exit. -c COLUMNS, --columns COLUMNS A comma separated list of column indices, names or ranges to sort by, e.g. "1,id,3-5". Defaults to all columns. -r, --reverse         Sort in descending order. -y SNIFF_LIMIT, --snifflimit SNIFF_LIMIT Limit CSV dialect sniffing to the specified number of bytes. Specify "
Potential document 3: csvsort_10: csvsort -c 9 examples/realdata/FY09_EDU_Recipients_by_State.csv
Potential document 4: csvsort_12: csvcut -c 1,9 examples/realdata/FY09_EDU_Recipients_by_State.csv | csvsort -r -c 2 | head -n 5
# sort a csv file by column 9
csvsort -c {{9}} {{data.csv}}

#END'''

tldr_0shot_prompt = '''Given the description, and some potential documents that might help, generate corresponding command. 
Only generate the command, and all variables in the command should be in the format of {{VAR}} and do not include "shell" or "bash".'''
# And pay attention that potential documents might not be helpful when generating the command

tldr_original_no_retrieval_prompt = '''# get the label of a fat32 partition
fatlabel {{/dev/sda1}}

#END

# display information without including the login, jcpu and pcpu columns
w --short

#END

# sort a csv file by column 9
csvsort -c {{9}} {{data.csv}}

#END'''

tldr_no_retrieval_prompt_with_instruction = '''Given the description, generate corresponding shell command, all variables in the command should be in the format of {{VAR}}
# get the label of a fat32 partition
fatlabel {{/dev/sda1}}

#END

# display information without including the login, jcpu and pcpu columns
w --short

#END

# sort a csv file by column 9
csvsort -c {{9}} {{data.csv}}

#END'''
