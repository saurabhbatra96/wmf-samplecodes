from extractor import Extractor

# Extraction code. To be shifted later.
# ext = Extractor('data/creditcard.csv')
# ext.fnfsplit(fraudpercent=0.5)
# ext.savedata('svmdata')
basedata = Extractor.loaddata('svmdata')
