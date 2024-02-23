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
You should be careful that potential documents might not contain the right information for generating the command
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


dummy_docs = '''potential document 0: dummy: The twirling twizzle spun and swirled, while the jolly jibber-jabber jangled joyfully. Tippy-tops tiptoed in the bumpy-boop, and snuggle-snacks snuggled sweetly. Doodle-dums dawdled dreamily in the dilly-dally, and wobble-wumps wavered whimsically. Zig-zags zipped through the zigzag zone, and snicker-snacks giggled gently. Wobble-waddles wandered and wiggled in the jibber-jabber jungle, lost in the muddle-puddle of nonsensical wizzle-wazzle. The flibber-jabber danced and twirled, while the zonk fizzled and popped. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Blork blork blork, gloopy gloopy gloopy. Wobble wobble wobble, zigzag zigzag zigzag. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quibble quibble quibble, jibber jabber jibber. Nulla facilisi. Snick snick snick, jibble jibble jibble.
potential document 1: dummy: The wiggly fluff went plop while the jibber-jabber bumbled and tumbled. Fizzle-flop danced around the wibbly-wobbly doodle, and snicker-snack bounced happily. Doodle-doo twirled and swirled in the zigzag zoom, and snuggle-bug snuggled close. Wobble-wobble wandered through the dilly-dally, giggling and jiggling all the while. Squiggle-squabble and waddle-waddle wobbled along, playing in the silly-sally world of random wozzle. The snickety-snack skipped and hopped, while the flibber-jabber giggled and squiggled. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Bloop bloop bloop, gloopy gloopy gloopy. Wobble wobble wobble, zigzag zigzag zigzag. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quibble quibble quibble, jibber jabber jibber. Nulla facilisi. Snick snick snick, jibble jibble jibble.
potential document 2: dummy: The merry muddle meandered and murmured, while the jovial jingle-jangle jiggled joyously. Tiptoe-tots tiptoed tenderly in the bouncy-bop, and snuggle-snacks snuggled softly. Doodle-daisies dallied dreamily in the dilly-dally, and wobble-wumps wavered whimsically. Zig-zag zephyrs zipped through the zigzag zone, and snicker-snacks snickered softly. Wobble-waddles wobbled and wiggled in the jibber-jabber jungle, lost in the whimsical wuzzle-wazzle. The blibble-blabble frolicked and twizzled, while the zonk zibber-zabbered and flubbled. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Blork blork blork, gloopy gloopy gloopy. Wobble wobble wobble, zigzag zigzag zigzag. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quibble quibble quibble, jibber jabber jibber. Nulla facilisi. Snick snick snick, jibble jibble jibble.
potential document 3: dummy: The zippity-zap zonked and ziggled, while the wibbly-wobbly wuzzle wandered and wiggled. Flippity-flop fluttered and floundered in the doodle-doo, and snicker-snack snuggled up. Dingle-dangle doddled and dawdled in the pitter-patter, and wobble-wabble wiggled away. Toodle-oo twirled and twiddled through the zigzag zoom, and snuggle-bug snickered softly. Wibble-wobble wandered and waddled in the jiggle-juggle jungle, lost in the riddle-raddle of nonsensical doodle-doo. The wobble-zonk danced and giggled, while the flibber-jabber snick-snacked and squiggled. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Bloop bloop bloop, gloopy gloopy gloopy. Wobble wobble wobble, zigzag zigzag zigzag. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quibble quibble quibble, jibber jabber jibber. Nulla facilisi. Snick snick snick, jibble jibble jibble 
potential document 4: dummy: The whimsical wuzzle wobbled and wiggled, while the dainty doodle danced delicately. Bumbling bips bounced in the bobbly-boop, and snuggle-snacks snuggled snugly. Doodle-dees dawdled dreamily in the dilly-dally, and wibble-wobbles wavered warmly. Tippy-toes tiptoed through the ticklish tangle, and snuggle-bugs snickered softly. Wobble-wopples wandered and waddled in the jibber-jabber jungle, lost in the muddle-puddle of nonsensical doodle-doo. The snickety-snick danced and twirled, while the flibber-jibber giggled and squiggled. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Bloop bloop bloop, gloopy gloopy gloopy. Wobble wobble wobble, zigzag zigzag zigzag. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quibble quibble quibble, jibber jabber jibber. Nulla facilisi. Snick snick snick, jibble jibble jibble.
potential document 5: dummy: The flibbertigibbet wobbled and zonked, while the quux fizzle quivered and snick snuck. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Blork blork blork, gloop gloop gloop. Wobble wobble wobble, zizzle zizzle zizzle. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quibble quibble quibble, flibber flibber flibber. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Snick snick snick, jibble jibble jibble. The wibble-wobble danced and giggled, while the flibber-jabber snick-snacked and squiggled. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Bloop bloop bloop, gloopy gloopy gloopy. Wobble wobble wobble, zigzag zigzag zigzag. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quibble quibble quibble, jibber jabber jibber. Nulla facilisi. Snick snick snick, jibble jibble jibble.
potential document 6: dummy: The zonk wobbled and quivered, while the fizzle snick snuck. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin ultrices dui vitae hendrerit mattis. Blork blork blork, gloop gloop gloop. Wobble wobble wobble, zizzle zizzle zizzle. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quibble quibble quibble, flibber flibber flibber. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Snick snick snick, jibble jibble jibble. The wobble-zonk frolicked and twizzled, while the flibber-jabber snick-snacked and squiggled. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Bloop bloop bloop, gloopy gloopy gloopy. Wobble wobble wobble, zigzag zigzag zigzag. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quibble quibble quibble, jibber jabber jibber. Nulla facilisi. Snick snick snick, jibble jibble jibble.
potential document 7: dummy: The zonk wobbled and quivered, while the fizzle snick snuck. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla facilisi. Blork blork blork, gloop gloop gloop. Wobble wobble wobble, zizzle zizzle zizzle. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quibble quibble quibble, flibber flibber flibber. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Snick snick snick, jibble jibble jibble. The flibber-jabber danced and giggled, while the zonk fizzled and popped. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Blork blork blork, gloopy gloopy gloopy. Wobble wobble wobble, zigzag zigzag zigzag. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quibble quibble quibble, jibber jabber jibber. Nulla facilisi. Snick snick snick, jibble jibble jibble.
potential document 8: dummy: The flibberjibber danced and giggled, while the zonk sizzled and fizzled. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla facilisi. Bloop bloop bloop, gloopy gloopy gloopy. Wobble wobble wobble, zigzag zigzag zigzag. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quibble quibble quibble, jibber jabber jibber. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Snick snick snick, jibble jibble jibble. The flibber-jibber danced and twirled, while the zonk fizzled and popped. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Blork blork blork, gloopy gloopy gloopy. Wobble wobble wobble, zigzag zigzag zigzag. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quibble quibble quibble, jibber jabber jibber. Nulla facilisi. Snick snick snick, jibble jibble jibble.
potential document 9: dummy: The snickity-snick danced through the zonk, while the quibble quivered in the flibber-jibber. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla facilisi. Bloop bloop bloop, gloopy gloopy gloopy. Wobble wobble wobble, zigzag zigzag zigzag. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quibble quibble quibble, jibber jabber jibber. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Snick snick snick, jibble jibble jibble. The zonk wobbled and quivered, while the fizzle snick snuck. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus feugiat urna id magna congue, vel cursus magna viverra. Blork blork blork, gloop gloop gloop. Wobble wobble wobble, zizzle zizzle zizzle. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quibble quibble quibble, flibber flibber flibber. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Snick snick snick, jibble jibble jibble.
'''