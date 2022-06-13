#!/usr/bin/env python                                                                                                                                                                                                       
import ROOT as r,sys,math,array,os
from optparse import OptionParser
from ROOT import std,RooDataHist
from array import array
import numpy as np
#from scipy.stats import poisson, norm, kstest, chi2
from scipy import stats
import sys

#resMass = 1900
#nominal = True


def end():
    if __name__ == '__main__':
        rep = ''
        while not rep in [ 'q', 'Q','a',' ' ]:
            rep = input( 'enter "q" to quit: ' )
            if 1 < len(rep):
                rep = rep[0]

def drawFrame(iX,iData,iFuncs,iCat):
    lCan   = r.TCanvas("qcd_"+iCat,"qcd_"+iCat,800,600)
    leg = r.TLegend(0.55,0.63,0.86,0.87)
    lFrame = iX.frame()
    lFrame.SetTitle("")
    lFrame.GetXaxis().SetTitle("m_{jj} (GeV)")
    lFrame.GetYaxis().SetTitle("Events")
    #iBkg.plotOn(lFrame,r.RooFit.FillColor(r.TColor.GetColor(100, 192, 232)),r.RooFit.FillStyle(3008), r.RooFit.DrawOption("E3"), r.RooFit.LineColor(r.kBlue))
    iData.plotOn(lFrame)
    iColor=51
    lRange = len(iFuncs)
    for i0 in range(lRange):
        if i0 == 1:
            iFuncs[i0].plotOn(lFrame,r.RooFit.LineColor(r.kGreen+1))
            chi_square = lFrame.chiSquare()
            print("CHI-SQUARED VALUE (Sig + Bkg): ", chi_square)
        else:
            iFuncs[i0].plotOn(lFrame,r.RooFit.LineColor(iColor),r.RooFit.LineStyle(r.kDashed))
            chi_square = lFrame.chiSquare()
            print("CHI-SQUARED VALUE (Bkg Only): ", chi_square)
        iColor+=10
    leg.SetFillColor(0)
    lFrame.Draw()
    lTmpData  = r.TH1F("tmpData" ,"tmpData" ,1,0,10); lTmpData .SetMarkerStyle(r.kFullCircle);
    #lTmpBkg   = r.TH1F("tmpBkg"  ,"tmpBkg"  ,1,0,10); lTmpBkg  .SetFillStyle(3008); lTmpBkg.SetLineColor(r.kBlue); lTmpBkg.SetFillColor(r.TColor.GetColor(100, 192, 232));
    lTmpFunc1 = r.TH1F("tmpFunc1","tmpFunc1",1,0,10); lTmpFunc1.SetLineColor(51);                lTmpFunc1.SetLineWidth(2); lTmpFunc1.SetLineStyle(r.kDashed);
    lTmpFunc2 = r.TH1F("tmpFunc2","tmpFunc2",1,0,10); lTmpFunc2.SetLineColor(61);                lTmpFunc2.SetLineWidth(2); lTmpFunc2.SetLineStyle(r.kDashed);
    lTmpFunc3 = r.TH1F("tmpFunc3","tmpFunc3",1,0,10); lTmpFunc3.SetLineColor(r.kGreen+1);        lTmpFunc3.SetLineWidth(2); #lTmpFunc3.SetLineStyle(r.kDashed);                                                             
    leg.AddEntry(lTmpData,"Toy Data","lpe")
    #leg.AddEntry(lTmpBkg ,"loss-sideband data","f")
    leg.AddEntry(lTmpFunc2,"Background-only Fit","lp")
    leg.AddEntry(lTmpFunc3,"Background+Signal Fit","lp")
    #leg.AddEntry(lTmpFunc1,"loss-sideband","lp")
    lPT = r.TPaveText(5200,20,6500,33)
    lPT.AddText("BlackBox 1")
    lPT.SetBorderSize(0)
    lPT.SetFillColor(0)
    leg.Draw()
    lCan.Modified()
    lCan.Update()
    lPT.Draw()
    #lCan.SaveAs(lCan.GetName()+".pdf")
    end()


def drawWorkspace(wspace, fitres, cutName, mh, nominal, sig):
    
    #myFile = r.TFile.Open("fileTEST.root", "RECREATE")
    myFile = r.TFile.Open("fileTEST.root", "UPDATE")
    lCan   = r.TCanvas("workspace","workspace",800,600)
    leg = r.TLegend(0.55,0.63,0.86,0.87)
    #lFrame = lWorkspace.data("data_obs").frame()
    lFrame = lWorkspace.var("x").frame()
    #lFrame = lWorkspace.var("CMS_channel").frame()
    #lFrame = iX.frame()
    fullName = cutName+"_"+str(mh)+"_sigma_"+str(sig)+"_nominal_"+str(nominal)
    lFrame.SetTitle(fullName)
    lFrame.GetXaxis().SetTitle("m_{jj} (GeV)")
    lFrame.GetYaxis().SetTitle("Events")
    #iBkg.plotOn(lFrame,r.RooFit.FillColor(r.TColor.GetColor(100, 192, 232)),r.RooFit.FillStyle(3008), r.RooFit.DrawOption("E3"), r.RooFit.LineColor(r.kBlue))
    #iData.plotOn(lFrame)
    lWorkspace.data("data_obs").plotOn(lFrame)
    iColor=51
    lWorkspace.var("MH").setVal(mh)
    lWorkspace.var("sigma").setVal(sig)
    
    #lWorkspace.var("a2_bkgL_140_10000_sigL_0_289").setVal(fitres.floatParsFinal().find("a2_bkgL_140_10000_sigL_0_289").getValV())
    #iter = fitres.floatParsFinal().createIterator();
    testw = fitres.getSnapshot("MultiDimFit")
    #testw.Print()
    iter = testw.createIterator()
    var = iter.Next()
    while var :
        if(var.GetName() != "w" and var.GetName() != "CMS_channel"):
            #print(var.GetName(), var.getVal())
            testvar = lWorkspace.var(var.GetName())
            if(testvar):
                #print(var.GetName())
                #print(var.getVal())
                lWorkspace.var(var.GetName()).setVal(var.getVal())
        var = iter.Next()

    """
    all_pars = fitres.floatParsFinal()
    #argvals = lWorkspace.allVars()
    for i in xrange(all_pars.getSize()):
        par = all_pars.at(i)
        #print(par.GetName())
        #if(par.GetName() == "sig_rate_bkgL_140_10000_sigL_0_289"):
        #if(par.GetName() == "sig_rate_bkgL_113_140_sigL_279_10000"):
        #if(par.GetName() == "sig_rate_bkgL_97_113_sigL_259_10000" and not nominal):
        if(par.GetName() == "sig_rate_"+cutName and not nominal):
            continue
        testvar = lWorkspace.var(par.GetName())
        if(testvar):
            print(par.GetName())
            print(par.getVal())
            lWorkspace.var(par.GetName()).setVal(par.getVal())
        #if(par.GetName() == "_ChannelCompatibilityCheck_r_bin_bkgL_140_10000_sigL_0_289"):
        #if(par.GetName() == "_ChannelCompatibilityCheck_r_bin_bkgL_113_140_sigL_279_10000"):
        #if(par.GetName() == "_ChannelCompatibilityCheck_r_bin_bkgL_97_113_sigL_259_10000" and not nominal):
        if(par.GetName() == "_ChannelCompatibilityCheck_r_bin_"+cutName and not nominal):
            print(par.GetName())
            print(par.getVal())
            #lWorkspace.var("sig_rate_bkgL_140_10000_sigL_0_289").setVal(par.getVal())
            #lWorkspace.var("sig_rate_bkgL_113_140_sigL_279_10000").setVal(par.getVal())
            #lWorkspace.var("sig_rate_bkgL_97_113_sigL_259_10000").setVal(par.getVal())
            lWorkspace.var("sig_rate_"+cutName).setVal(par.getVal())
    """
    #lWorkspace.pdf("pdf_binbin_bkgL_140_10000_sigL_0_289_nuis").plotOn(lFrame)
    #lWorkspace.pdf("pdf_binbin_bkgL_113_140_sigL_279_10000_nuis").plotOn(lFrame)
    #lWorkspace.pdf("pdf_binbin_bkgL_97_113_sigL_259_10000_nuis").plotOn(lFrame)
    lWorkspace.pdf("pdf_binbin_"+cutName+"_nuis").plotOn(lFrame)
    #fitres.plotOn(lFrame)
    lFrame.Draw()
    #print(cutName, lFrame.chiSquare())
    #print(cutName, lFrame.makeResidHist())
    #print(lFrame.residHist()._entries)
    hresid = lFrame.residHist()
    #frame2 = lWorkspace.var("x").frame(Title("Residual Distribution"))
    frame2 = lWorkspace.var("x").frame()
    frame2.SetTitle("resids"+cutName)
    frame2.GetXaxis().SetTitle("m_{jj} (GeV)")
    frame2.GetYaxis().SetTitle("Events")
    frame2.addPlotable(hresid,"P")
    frame2.Draw()
    """
    lRange = len(iFuncs)
    for i0 in range(lRange):
        if i0 == 1:
            iFuncs[i0].plotOn(lFrame,r.RooFit.LineColor(r.kGreen+1))
            chi_square = lFrame.chiSquare()
            print("CHI-SQUARED VALUE (Sig + Bkg): ", chi_square)
        else:
            iFuncs[i0].plotOn(lFrame,r.RooFit.LineColor(iColor),r.RooFit.LineStyle(r.kDashed))
            chi_square = lFrame.chiSquare()
            print("CHI-SQUARED VALUE (Bkg Only): ", chi_square)
        iColor+=10
    leg.SetFillColor(0)
    lFrame.Draw()
    lTmpData  = r.TH1F("tmpData" ,"tmpData" ,1,0,10); lTmpData .SetMarkerStyle(r.kFullCircle);
    #lTmpBkg   = r.TH1F("tmpBkg"  ,"tmpBkg"  ,1,0,10); lTmpBkg  .SetFillStyle(3008); lTmpBkg.SetLineColor(r.kBlue); lTmpBkg.SetFillColor(r.TColor.GetColor(100, 192, 232));
    lTmpFunc1 = r.TH1F("tmpFunc1","tmpFunc1",1,0,10); lTmpFunc1.SetLineColor(51);                lTmpFunc1.SetLineWidth(2); lTmpFunc1.SetLineStyle(r.kDashed);
    lTmpFunc2 = r.TH1F("tmpFunc2","tmpFunc2",1,0,10); lTmpFunc2.SetLineColor(61);                lTmpFunc2.SetLineWidth(2); lTmpFunc2.SetLineStyle(r.kDashed);
    lTmpFunc3 = r.TH1F("tmpFunc3","tmpFunc3",1,0,10); lTmpFunc3.SetLineColor(r.kGreen+1);        lTmpFunc3.SetLineWidth(2); #lTmpFunc3.SetLineStyle(r.kDashed);                                                             
    leg.AddEntry(lTmpData,"Toy Data","lpe")
    #leg.AddEntry(lTmpBkg ,"loss-sideband data","f")
    leg.AddEntry(lTmpFunc2,"Background-only Fit","lp")
    leg.AddEntry(lTmpFunc3,"Background+Signal Fit","lp")
    #leg.AddEntry(lTmpFunc1,"loss-sideband","lp")
    lPT = r.TPaveText(5200,20,6500,33)
    lPT.AddText("BlackBox 1")
    lPT.SetBorderSize(0)
    lPT.SetFillColor(0)
    leg.Draw()
    """
    #lCan.Modified()
    #lCan.Update()
    #lPT.Draw()
    #myFile.WriteObject(lCan, "MyCan")
    myFile.WriteObject(lFrame, fullName)
    myFile.WriteObject(frame2, fullName+"resids")
    #lCan.SaveAs(lCan.GetName()+".pdf")
    #end()
    #print(lFrame.GetNbinsX())
    return lFrame.chiSquare()*lFrame.GetNbinsX()


if __name__ == "__main__":
    #lTFile = r.TFile("./fullCard.root")
    #lTFile = r.TFile("./test_bkgL_140_10000_sigL_0_289.root")
    #lTFile = r.TFile("./test_bkgL_113_140_sigL_279_10000.root")
    r.gSystem.Load('RooMyPdf_cxx.so')
    r.gROOT.SetBatch(r.kTRUE)

    """
    i = 1;
    #if sys.argv[i] == "combine": i+= 1
    while i < len(sys.argv):
        if sys.argv[i] == "-m":
            resMass = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "-n":
            nominal = bool(sys.argv[i+1])
            i += 2
        else:
            print("Invalid input")
            i += 1
    """

    minSigma = SIGMAMIN
    maxSigma = SIGMAMAX
    gapSigma = SIGMAGAP
    cutNameList = [LIST]
    #nominalPossibilities = [True, False]
    nominalPossibilities = ['Sig','NoSig']
    minM = MINIMUMMASS
    maxM = MAXIMUMMASS
    spacing = GAPS

    

    relevantIndices = []
    for i in range(len(cutNameList)):
        print(i, i%np.sqrt(len(cutNameList)), math.floor(float(i)/np.sqrt(len(cutNameList))), np.sqrt(len(cutNameList)))
        if(i%np.sqrt(len(cutNameList)) == 0 and math.floor(float(i)/np.sqrt(len(cutNameList))) == np.sqrt(len(cutNameList))-1):
            relevantIndices.append(i)
        if(i%np.sqrt(len(cutNameList)) == 0 and math.floor(float(i)/np.sqrt(len(cutNameList))) == np.sqrt(len(cutNameList))-2):
            relevantIndices.append(i)
        if(i%np.sqrt(len(cutNameList)) == 1 and math.floor(float(i)/np.sqrt(len(cutNameList))) == np.sqrt(len(cutNameList))-1):
            relevantIndices.append(i)

    #cutName = "bkgL_97_113_sigL_259_10000"
    #pvalsSig = []
    for sig in range(minSigma, maxSigma+gapSigma, gapSigma):
        plottablePs = []
        relevantPs = []
        plottableMasses = []
        for resMass in range(minM, maxM+spacing, spacing):
            pvals = []
            for cutName in cutNameList:
                lTFile = r.TFile("./test_"+cutName+".root")
                lWorkspace = lTFile.Get("w")
                
                #file2 = r.TFile("./higgsCombineTest.ChannelCompatibilityCheck.mH3150.root")
                #for resMass in range(minM, maxM, spacing):
                #for resMass in [2000,2500,2800,3200,3700]:
                chisqs = []
                for nominal in nominalPossibilities:
                    file2 = r.TFile("./higgsCombine"+nominal+".MultiDimFit.mH"+str(resMass)+"_sigma"+str(sig)+".root")
                    lFitRes = file2.Get("w")
                    chisqs.append(drawWorkspace(lWorkspace, lFitRes, cutName, resMass, nominal, sig))
                #for nominal in nominalPossibilities:
                #    if(nominal):
                #        lFitRes = file2.Get("w")
                #    else:
                #        lFitRes = file2.Get("w")
                #    
                #    #lWorkspace.Print()
                #    if(lFitRes):
                #        drawWorkspace(lWorkspace, lFitRes, cutName, resMass, nominal, sig)
                print(chisqs)
                print(cutName, stats.chi2.sf(chisqs[1]-chisqs[0],1))
                pvals.append(stats.chi2.sf(chisqs[1]-chisqs[0],1))
                lTFile.Close()
            logp = 0.
            for p in pvals:
                logp = logp - 2.*np.log(p)
            logpR = 0.
            for rel in relevantIndices:
                print("CHECK OUT RELEVANT BINS?", rel, pvals[rel])
                logpR = logpR - 2.*np.log(pvals[rel])
            print("RESMASS IS ", resMass, "SIGMA IS", sig, "PVAL IS", stats.chi2.sf(logp,2*len(pvals)))
            plottablePs.append(stats.chi2.sf(logp,2*len(pvals)))
            relevantPs.append(stats.chi2.sf(logpR,2*len(relevantIndices)))
            plottableMasses.append(resMass)
            #end()
            #logp = 0.
            #for p in pvals:
            #    logp = logp - 2.*np.log(p)
            #print(stats.chi2.sf(logp,2*len(pvals)))
            
        #lTFile = r.TFile("./fullMassScan_sigma"+str(sig)+".root")
        #myTree = lTFile.Get("limit")
        
        #masses = []
        #chi2 = []
        masses, chi2, pvalues, pRvalues = array( 'd' ), array( 'd' ), array( 'd' ), array( 'd' )
        
        n=0
        #for entry in myTree:
        for e in range(len(plottableMasses)):
            masses.append(plottableMasses[e])
            pvalues.append(plottablePs[e])
            pRvalues.append(relevantPs[e])
            #print(str(entry.mh)+" chi2 = "+str(entry.limit)+" pvalue 9 = "+str(1. - stats.chi2.cdf(entry.limit,9))+" pvalue 8 = "+str(1. - stats.chi2.cdf(entry.limit,8)))
            #print(str(entry.mh)+" chi2 = "+str(entry.limit)+" pvalue 9 = "+str(stats.chi2.sf(entry.limit,9))+" pvalue 8 = "+str(stats.chi2.sf(entry.limit,8)))
            #pvalues.append(stats.chi2.sf(entry.limit,len(cutNameList)))
            n+=1
            
        #gr = r.TGraph( n, masses, chi2 )
        #gr.SetMarkerColor( 4 )
        #gr.SetMarkerStyle( 21 )
        #gr.SetTitle( 'chi2 vs resMass for sigma '+str(sig) )
        #gr.GetXaxis().SetTitle( 'Mass Hypothesis' )
        #gr.GetYaxis().SetTitle( 'Pseudo Chi2' )
        #gr.Draw()
        
        
        lC0 = r.TCanvas("mh_scan_sigma_"+str(sig),"mh_scan_sigma_"+str(sig),800,600)
        leg = r.TLegend(0.55,0.23,0.86,0.47)
        leg.SetFillColor(0)
        lGraphs=[]
        sigmas=[]
        
        graph1 = r.TGraph(n,masses,pvalues)
        graph1.SetMarkerStyle(20)
        graph1.GetXaxis().SetTitle("m_{jj} (GeV)")
        graph1.GetYaxis().SetTitle("p^{0} value")
        graph1.SetTitle("")#Significance vs Mass")                                                                                                                                                                          
        #graph1.SetLineColor(51+i0*12)
        #graph1.SetMarkerColor(51+i0*12)
        #graph1.SetLineWidth(2+i0)
        r.gPad.SetLogy(True)
        #graph1.Draw()
        graph1.Draw("alp")
        lGraphs.append(graph1)
        leg.AddEntry(graph1,'test',"lp")
        lines=[]
        for i0 in range(20):#len(sigmas)):
            #print(1-stats.norm.cdf(i0+1), stats.norm.sf(i0+1))
            #sigmas.append(1-stats.norm.cdf(i0+1))
            sigmas.append(stats.norm.sf(i0+1))
            lLine = r.TLine(masses[0],sigmas[i0],masses[len(masses)-1],sigmas[i0])
            lLine.SetLineStyle(r.kDashed)
            lLine.SetLineWidth(2)
            lLine.Draw()
            #lPT = r.TPaveText(3200,sigmas[i0],3700,sigmas[i0]+1.5*sigmas[i0])
            lPT = r.TPaveText(3200,sigmas[i0],3700,sigmas[i0]+20.5*sigmas[i0])
            lPT.SetFillStyle(4050)
            lPT.SetFillColor(0)
            lPT.SetBorderSize(0)
            lPT.AddText(str(i0+1)+"#sigma")
            lPT.Draw()
            lines.append(lLine)
            lines.append(lPT)
            
        for pGraph in lGraphs:
            pGraph.Draw("lp")
        #leg.Draw()
        lC0.Update()
        lC0.Draw()
        
        
        
        lCR = r.TCanvas("mh_relevant_scan_sigma_"+str(sig),"mh_relevant_scan_sigma_"+str(sig),800,600)
        legR = r.TLegend(0.55,0.23,0.86,0.47)
        legR.SetFillColor(0)
        lGraphs2=[]
        sigmas2=[]
        
        graph2 = r.TGraph(n,masses,pRvalues)
        graph2.SetMarkerStyle(20)
        graph2.GetXaxis().SetTitle("m_{jj} (GeV)")
        graph2.GetYaxis().SetTitle("p^{0} value")
        graph2.SetTitle("")#Significance vs Mass")                                                                                                                                                                          
        #graph1.SetLineColor(51+i0*12)
        #graph1.SetMarkerColor(51+i0*12)
        #graph1.SetLineWidth(2+i0)
        r.gPad.SetLogy(True)
        #graph1.Draw()
        graph2.Draw("alp")
        lGraphs2.append(graph2)
        legR.AddEntry(graph2,'test2',"lp")
        lines2=[]
        for i0 in range(20):#len(sigmas)):
            #print(1-stats.norm.cdf(i0+1), stats.norm.sf(i0+1))
            #sigmas.append(1-stats.norm.cdf(i0+1))
            sigmas2.append(stats.norm.sf(i0+1))
            lLine = r.TLine(masses[0],sigmas2[i0],masses[len(masses)-1],sigmas2[i0])
            lLine.SetLineStyle(r.kDashed)
            lLine.SetLineWidth(2)
            lLine.Draw()
            #lPT = r.TPaveText(3200,sigmas[i0],3700,sigmas[i0]+1.5*sigmas[i0])
            lPT = r.TPaveText(3200,sigmas2[i0],3700,sigmas2[i0]+20.5*sigmas2[i0])
            lPT.SetFillStyle(4050)
            lPT.SetFillColor(0)
            lPT.SetBorderSize(0)
            lPT.AddText(str(i0+1)+"#sigma")
            lPT.Draw()
            lines2.append(lLine)
            lines2.append(lPT)
            
        for pGraph in lGraphs2:
            pGraph.Draw("lp")
        #graph2.Draw()
        #for pGraph in lGraphs:
        #    pGraph.Draw("lp")
        #leg.Draw()
        lCR.Update()
        lCR.Draw()
        
        
        
        
        myFile = r.TFile.Open("fileTEST.root", "UPDATE")
        #myFile.WriteObject(gr, 'chi2_vs_resMass_sigma'+str(sig))
        myFile.WriteObject(graph1, 'pval_vs_resMass_sigma'+str(sig))
        myFile.WriteObject(lC0, 'SignificancePlot_sigma'+str(sig))
        myFile.WriteObject(lCR, 'RelevantSignificancePlot_sigma'+str(sig))
        myFile.Close()
        #lCan   = r.TCanvas("workspace","workspace",800,600)
        
        
        #myTree.Draw("limit:mh")
        #lCan.Modified()
        #lCan.Update()
        #myFile.WriteObject(lCan, "MyCan")        
