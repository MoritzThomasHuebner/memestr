EVENT_LIST="GW190413B GW190421A GW190424A GW190426A GW190503A GW190512A GW190513A GW190514A GW190517A GW190519A GW190521 GW190521A"

for EVENT in ${EVENT_LIST}
do
  bilby_pipe ${EVENT}.ini
done
