from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.cell(190, 10, 'Savunma Sanayi ve Havacilik Odakli Derin Ogrenme Yol Haritasi', align='C')
        self.ln(15)

    def chapter_title(self, title, num):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(190, 10, f'CHAPTER {num}: {title}', align='L', fill=True)
        self.ln(12)

    def chapter_body(self, steps):
        self.set_font('Helvetica', '', 10)
        for step in steps:
            self.cell(190, 8, step, align='L')
            self.ln(8)
        self.ln(10)

pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

chapter_3_title = "Siniflandirma ve IHA Anomali Tespiti"
chapter_3_steps = [
    "Aga sayi tahmin etmenin (Regresyon) otesinde, kategorize etme yetenegi kazandirilmasi.",
    "",
    "ADIM 3.1 -> Teorik Temel: Softmax Aktivasyonu",
    "ADIM 3.2 -> Loss Fonksiyonu: Categorical Cross-Entropy (CCE)",
    "ADIM 3.3 -> Veri Isleme: One-Hot Encoding",
    "ADIM 3.4 -> Degerlendirme: Classification Metrics (Accuracy, Confusion Matrix)",
    "ADIM 3.5 -> Veri Uretimi: Sentetik IHA Sensor Verisi Simulatoru",
    "ADIM 3.6 -> Model Guncellemesi: basic_neural_training.py modulunun uyarlanmasi",
    "ADIM 3.7 -> FINAL PROJESI: IHA Sensor Arizasi ve GPS Spoofing Tespit Sistemi",
    "ADIM 3.8 -> Dokumantasyon: GitHub ve README guncellemeleri"
]

pdf.chapter_title(chapter_3_title, 3)
pdf.chapter_body(chapter_3_steps)


chapter_4_title = "Zaman Serileri ve Otopilot Davranis Klonlama"
chapter_4_steps = [
    "Aga 'gecmisi hatirlama' yetenegi kazandirarak, ucagin hamlesini tahmin eden altyapi.",
    "",
    "ADIM 4.1 -> Teorik Temel: Zaman Serileri (Time-Series) Mantigi",
    "ADIM 4.2 -> Veri Isleme: Sliding Window (Kayan Pencere) Teknigi",
    "ADIM 4.3 -> Simulasyon Kurulumu: ArduPilot / Mission Planner (SITL) Entegrasyonu",
    "ADIM 4.4 -> Veri Toplama: Simulasyondan Telemetri (.tlog / .bin) Cekilmesi",
    "ADIM 4.5 -> Model Mimarisi: Kinematik verilere uygun ag yapisinin kurulmasi",
    "ADIM 4.6 -> FINAL PROJESI: Behavioral Cloning - Ucagin Elevator/Aileron manevralarini tahmin.",
    "ADIM 4.7 -> Dokumantasyon: GitHub ve README guncellemeleri",
    "",
    "NOT: Her bir adimda, ilgili kodlari mevcut kutuphanene ekleyerek ilerleyecegiz."
]

pdf.chapter_title(chapter_4_title, 4)
pdf.chapter_body(chapter_4_steps)

pdf_path = r"C:\Users\24406601057\Desktop\Deep Learning\pdf\Yol_Haritasi.pdf"
pdf.output(pdf_path)
print(f"PDF successfully created at {pdf_path}")
