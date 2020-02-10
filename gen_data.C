// Generate a data set and a background set
// Background is a random exponential and data is an exponential + a little gaussian bump

void gen_data(){
	//Create file to store trees
	TFile* Ofile = TFile::Open("data.root","RECREATE");
	
	//Create the trees
	TTree* Dtree = new TTree("data","data");
	TTree* Btree = new TTree("bkg","bkg");
	
	//Create 2 branches (bkg and data)
	double data,bkg;
	Btree->Branch("bkg",&bkg,"bkg/D");
	Dtree->Branch("data",&data,"data/D");
	
	//Random generator
	TRandom* G = new TRandom(42);
	
	//Fill them
	for(int i=0;i<100000;i++){
		data = G->Exp(2);
		bkg = G->Exp(2);
		if(bkg<20.0){
			Btree->Fill();
		}
		if(data<20.0){
			Dtree->Fill();
		}
	}
	for(int i=0;i<300;i++){
		data = G->Gaus(5.5,0.35);
		if(data<20.0){
			Dtree->Fill();
		}
	}
	
	//Save them in file
	Ofile->Write("bkg");
	Ofile->Write("data");
	
	
	//Create new file to store histograms
	TFile* Ofile2 = TFile::Open("hist.root","RECREATE");
	Ofile2->cd();
	
	//Create histograms from the trees
	TH1F* Hdata = new TH1F("data_dijet","data_dijet",60,0,20);
	TH1F* Hbkg = new TH1F("bkg_dijet","bkg_dijet",60,0,20);
	Dtree->Draw("data>>data_dijet(60,0,20)","");
	Btree->Draw("bkg>>bkg_dijet(60,0,20)","");
	
	//save them in new file
	Ofile2->Write("data_dijet");
	Ofile2->Write("bkg_dijet");
	/**/
	//Close files
	Ofile->Close();
	Ofile2->Close();
}



