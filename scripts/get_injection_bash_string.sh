#!/usr/bin/env bash
FILENAME="./parameter_sets/$1"
while IFS= read -r var
do
  echo "$var"
done < "$FILENAME"
