#!/bin/bash
# Update the environment.yml file with the current list of packages, e.g., after a fix

conda env export --name rl2  > environment.yml
