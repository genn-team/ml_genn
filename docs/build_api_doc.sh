#! /bin/bash
sphinx-apidoc --maxdepth 1 --module-first --force -o source/ ../ml_genn
sphinx-apidoc --maxdepth 1 --module-first --force -o source/ ../ml_genn_tf