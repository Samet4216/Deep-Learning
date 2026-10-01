from fpdf import FPDF
import shutil
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 13)
        self.cell(190, 8, 'Savunma Sanayi ve Havacilik Odakli Derin Ogrenme Yol Haritasi', align='C')
        self.ln(10)

    def chapter_title(self, title, num):
        self.set_font('Helvetica', 'B', 10.5)
        self.set_fill_color(210, 230, 255)
        self.cell(190, 8, f'CHAPTER {num}: {title}', align='L', fill=True)
        self.ln(9)

    def chapter_body(self, steps):
        self.set_font('Helvetica', '', 8.5)
        for step in steps:
            if step.startswith("ADIM"):
                self.set_font('Helvetica', 'B', 8.5)
                self.cell(190, 5, step, align='L')
                self.set_font('Helvetica', '', 8.5)
            elif step.startswith("  -"):
                self.set_font('Helvetica', '', 7.5)
                self.cell(190, 4.5, step, align='L')
                self.set_font('Helvetica', '', 8.5)
            elif step == "":
                self.ln(1.5)
                continue
            else:
                self.set_font('Helvetica', 'I', 8)
                self.cell(190, 4.5, step, align='L')
                self.set_font('Helvetica', '', 8.5)
            self.ln(5.2)
        self.ln(4)

pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=10)
pdf.add_page()

# CHAPTER 3
chapter_3_title = "Siniflandirma ve IHA Siber/Sensor Anomali Tespiti (Ileri Seviye)"
chapter_3_steps = [
    "Kutuphanemizi sayisal tahminden cok-sinifli karar verme mekanizmasina donusturuyoruz.",
    "",
    "ADIM 3.1 -> Teorik Temel: Softmax Aktivasyonu ve Sayisal Kararlilik (Max-Shift)",
    "ADIM 3.2 -> Loss Fonksiyonu: Categorical Cross-Entropy (CCE) ve Bilgi Kurami",
    "ADIM 3.3 -> Matematiksel Ispat: Softmax + CCE Geri Yayilim Turevi (dZ = A - Y)",
    "  - Jacobian matrisi ve zincir kuralinin analitik olarak sadelesme ispati",
    "ADIM 3.4 -> Karsilastirma: Ikili (Binary / BCE) vs. Coklu (Multi-Class / CCE) Mimarisi",
    "ADIM 3.5 -> Veri Isleme: One-Hot Encoding ve Kategorik Donusum Algoritmasi",
    "ADIM 3.6 -> Dengesiz Veri (Class Imbalance) ve Agirlikli Kayip (Weighted CCE)",
    "  - Savunma sanayiinde azinlik siniflarin (saldiri/ariza) kacirilmasini onleme",
    "ADIM 3.7 -> Modern Duzenlilestirme (Regularization): Label Smoothing Mekanizmasi",
    "  - Modelin asiri ozguvenli (overconfident) olmasini ve ezberini engelleme",
    "ADIM 3.8 -> Ileri Seviye Metrikler: Confusion Matrix, Precision, Recall ve F1-Score",
    "  - Neden harmonik ortalama kullanilir? Macro vs. Weighted metrik analizleri",
    "ADIM 3.9 -> Gorsellestirme: 2D Karar Sinirlarinin (Decision Boundary) Cizdirilmesi",
    "  - Agin siniflari birbirinden nasil ayirdigini kontur haritasiyla inceleme",
    "ADIM 3.10 -> Veri Uretimi: Fizik Tabanli Sentetik IHA Sensor Simulatoru",
    "ADIM 3.11 -> Model Entegrasyonu: basic_neural_training.py'nin Cok Amacli Yapilandirilmasi",
    "  - Regresyon ve Siniflandirma gorevlerini ayni cati altinda calistirma",
    "ADIM 3.12 -> FINAL PROJESI: Gercek Zamanli IHA Sensor Arizasi ve GPS Spoofing Tespiti",
    "ADIM 3.13 -> Dokumantasyon: Kapsamli GitHub ve LinkedIn Raporlamasi"
]

pdf.chapter_title(chapter_3_title, 3)
pdf.chapter_body(chapter_3_steps)

# CHAPTER 4 (EXTENDED & COMPREHENSIVE)
chapter_4_title = "Zaman Serileri ve Otopilot Davranis Klonlama (RNN & BPTT Mimarisi)"
chapter_4_steps = [
    "Aga 'hafiza' kazandirarak, ucagin gecmis hareketlerine gore otopilot manevrasi kestirimi.",
    "",
    "ADIM 4.1 -> Zaman Serileri Temelleri: Duraganlik, Trend, Gecikme (Lag) ve Otokorelasyon",
    "  - Standart MLP aglarinin zaman boyutunda neden yetersiz kaldiginin analizi",
    "ADIM 4.2 -> Zamansal Veri Bolme (Temporal Split) ve Data Leakage Korumasi",
    "  - Zaman serilerinde 'random split' yapilamaz! Kronolojik bolme kurali",
    "ADIM 4.3 -> 3D Tensor Donusumu ve Vektorize Sliding Window (Kayan Pencere) Algoritmasi",
    "  - [Batch, Features] yapisindan [Batch, Timesteps/Lookback, Features] yapisina gecis",
    "ADIM 4.4 -> Hafizali Ag Mimarisi: Vanilla RNN ve Gizli Durum (Hidden State - h_t) Anatomisi",
    "  - Agirlik paylasimi (Weight Sharing across time) ve tanh aktivasyonu felsefesi",
    "ADIM 4.5 -> Matematiksel Zirve: Zamanda Geri Yayilim (BPTT - Backprop Through Time)",
    "  - Agin zamanda acilmasi (Unrolling) ve zincir kuraliyla analitik turev ispati",
    "ADIM 4.6 -> Dinamik Kararlilik: Vanishing / Exploding Gradients ve Gradient Clipping",
    "  - Tekrarli matris carpimlarinda gradyanlari norm bazli traslayarak patlamayi onleme",
    "ADIM 4.7 -> Ileri Seviye Hucre Mimarilerine Bakis: LSTM (Long Short-Term Memory) Anatomisi",
    "  - Uzun vadeli bagimliliklar, Forget/Input/Output Gate ve Hucre Durumu (C_t)",
    "ADIM 4.8 -> Zaman Serisi Metrikleri: Gecikme (Phase Shift) Hatasi ve Yon Dogrulugu",
    "ADIM 4.9 -> Simulasyon Ortami: ArduPilot SITL / Mission Planner Kurulumu",
    "ADIM 4.10 -> Telemetri Veri Pipeline'i: MAVLink (.tlog / .bin) Ucus Loglarindan Veri Cekme",
    "  - IMU (Ivmeolcer, Jiroskop), GPS ve Pitot tupu sensorlerinin senkronize edilmesi",
    "ADIM 4.11 -> RNN Modul Entegrasyonu: Kutuphanemize RNNLayer Sinifinin Eklenmesi",
    "ADIM 4.12 -> FINAL PROJESI: Behavioral Cloning - Yapay Zeka Destekli Otopilot Manevra Kestirimi",
    "  - Ucagin gecmis 3 saniyelik kinematigine bakarak Aileron/Elevator komutlarini uretme",
    "ADIM 4.13 -> Havacilik Testi: Gercek Ucus Verileri ile Model Tahminlerinin Karsilastirilmasi",
    "ADIM 4.14 -> Dokumantasyon: Kapsamli Teknik Makale, Havacilik Grafikleri ve GitHub Raporu"
]

pdf.chapter_title(chapter_4_title, 4)
pdf.chapter_body(chapter_4_steps)

# Save to project pdf directory
pdf_proj_path = r"C:\Users\24406601057\Desktop\Deep Learning\pdf\Yol_Haritasi.pdf"
pdf.output(pdf_proj_path)
print(f"Project PDF created: {pdf_proj_path}")

# Copy to Downloads directory
downloads_path = os.path.expanduser(r"~\Downloads\Yol_Haritasi.pdf")
shutil.copy(pdf_proj_path, downloads_path)
print(f"Copied to Downloads: {downloads_path}")
