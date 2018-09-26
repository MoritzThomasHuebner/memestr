import memestr
import sys


id = sys.argv[1]
sys.stdout.write(memestr.submit.submitter.get_injection_bash_strings(id=id))
sys.stdout.flush()
