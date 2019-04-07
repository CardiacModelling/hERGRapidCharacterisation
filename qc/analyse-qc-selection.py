#!/usr/bin/env python2
import sys
sys.path.append('../lib')
import pickle
import hergqc

folder = '.'
ID = 'herg25oc1'
prt = 'staircaseramp'

SELECTION = pickle.load(open('%s/%s-%s-all_qc.p' % (folder, ID, prt), 'rb'))

QCnames = hergqc.hERGQC().qc_names()

saveas = '%s/%s-%s-qcvisual' % (folder, ID, prt)

hergqc.visual_hergqc(SELECTION, ID, QCnames, saveas)

