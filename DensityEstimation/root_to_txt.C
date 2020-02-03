
std::string dir_path = "C:\\Users\\Adi\\Documents\\Faks\\semestar_7\\CUDA\\DensityEstimation\\DensityEstimation\\data\\in\\";
std::string file_name = "toyXic2pKpi_200k_KstarAmp20percentage";

TFile file((dir_path + file_name + ".root").c_str());
TTree* tree = static_cast<TTree*>(file.Get("R0Tree"));

TTreeReader reader("R0Tree", &file);
TTreeReaderValue<float> m2pk(reader, "m2pk");
TTreeReaderValue<float> m2kpi(reader, "m2kpi");
TTreeReaderValue<int> id(reader, "id");

std::ofstream out_file(dir_path + file_name + ".txt");
out_file << tree->GetEntries() << std::endl;

while (reader.Next()) {
    out_file << *id << ' ' << *m2pk << ' ' << *m2kpi << std::endl;
}

out_file.close();
