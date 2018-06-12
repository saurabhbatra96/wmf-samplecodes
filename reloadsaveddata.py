from extractor import Extractor

ext = Extractor('data/creditcard.csv')
ext.fnfsplit(fraudpercent=0.5)
ext.savedata('data-3-0.5')