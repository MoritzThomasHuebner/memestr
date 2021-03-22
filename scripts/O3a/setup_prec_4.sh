EVENT_LIST="GW190408A_prec GW190707A_prec GW190708A_prec GW190720A_prec GW190728A_prec"
for EVENT in ${EVENT_LIST}
do
  bilby_pipe ${EVENT}.ini
done
