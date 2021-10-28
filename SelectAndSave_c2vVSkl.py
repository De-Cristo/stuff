from root_numpy import root2array, rec2array, array2root
import sys
import os

#A simple skim script, only keep the information needed for fast training 

def selectandsave(filestring, outstring, passcut, branch_names):
    test = root2array(filestring, "Events", branches=branch_names, selection=passcut)
    array2root(test, outstring,treename='Events', mode='recreate')


branch_names = [
            'VHH_H1_BJet1_btag','VHH_H1_BJet2_btag','VHH_H2_BJet1_btag','VHH_H2_BJet2_btag',
            'VHH_H1_m','VHH_H1_pT','VHH_H1_eta','VHH_H1_phi','VHH_H1_e',
            'VHH_H2_m','VHH_H2_pT','VHH_H2_eta','VHH_H2_phi','VHH_H2_e',
            'VHH_HH_m','VHH_HH_pT','VHH_HH_eta','VHH_HH_phi','VHH_HH_e',
            'VHH_HH_dphi','VHH_HH_deta','VHH_HH_dR','VHH_rHH','VHH_HT',
            'VHH_mass','VHH_Vreco4j_HT','VHH_vsptmassRatio','VHH_H2H1_pt_ratio','VHH_V_HH_dPhi','VHH_V_H1_dPhi','VHH_V_H2_dPhi',
            'VHH_V_phi','VHH_V_m','VHH_V_e','VHH_nBJets','VHH_nBJets_loose','VHH_nBJets_tight','MET_Pt','MET_Phi','V_pt',
            'weight','isZnn','isZmm','isZee','selLeptons_pt_0','selLeptons_phi_0','selLeptons_eta_0',
            'isWenu','isWmunu','lepMetDPhi'
            ]

Filelist = ['CV_1_0_C2V_0_0_C3_1_0',
            'CV_0_5_C2V_1_0_C3_1_0',
            'CV_1_0_C2V_1_0_C3_0_0',
            'CV_1_0_C2V_1_0_C3_1_0',
            'CV_1_0_C2V_1_0_C3_2_0',
            'CV_1_0_C2V_2_0_C3_1_0',
            'CV_1_5_C2V_1_0_C3_1_0']

for ifile in Filelist:
    passcut='isResolved'
    filestring='../TEST_0919_BDT/*'+ifile+'*/output_*.root'
    outstring=ifile+'_skim.root'
    selectandsave(filestring, outstring, passcut, branch_names)





