# LitMonitor

A simple, lightweight agent that evaluates new PubMed articles for you. Based on LangGraph.

## New Lit Monitor

Non-interactive python script that performs a single monitor run based on a JSON configuration file.

new_lit_monitor.py [-h] [-U] [-O OUTPUT] [-K KEY] filename

Check the relevance of the most recent papers matching a PubMed search. Default behavior is to ignore any list of previously-seen articles in the input file and to output results to a new file with the same name as the input file plus a timestamp. If --update is used, previously-seen articles in the input file are NOT reevaluated, and the output results will be combined with the input file and written to that file. The --output option allows one to specify a name for the output file, regardless of whether the --update flag is set.

positional arguments:
  filename              Path to the configuration file to use

options:
  -h, --help            show this help message and exit
  -U, --update          Update the previously-run search described in the configuration file.
  -O OUTPUT, --output OUTPUT
                        Path to the output file to store results.
  -K KEY, --key KEY     API key to use with the LLM, if needed.

## Eval New Lit Monitor

Evaluate the performance of a specific configuration of the new_lit_monitor agent by testing it on a list of articles where we have provided gold-standard human labels.

positional arguments
  eval_file   Path to a JSON settings file where the articles have been categorized by a human.

options:
  -R REPEATS, --repeats REPEATS
                        Optional flag to run the agent multiple times for each article in the evaluation file.
  -O OUTPUT, --output OUTPUT
                        Path to the output file to store results.
  -K KEY, --key KEY     API key to use with the LLM, if needed.

## GUI New Lit Eval

User interface based on NiceGUI that displays the results of a monitor run.

- Currently only works with evaluation results.
- Update to allow missing gold standard results, which can be then be filled in by the user.

## GUI A/B Eval Comparison

User interface based on NiceGUI that allows comparing two runs on the same set of articles. Intended for A/B testing, to determine whether a settings change has improved or worsened the monitor results.

- Not working yet.
