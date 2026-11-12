from dataclasses import dataclass
import random
from typing import Dict, List


VALID_EXPRESSIONS = ("menguap", "minum", "istirahat", "olahraga", "jalan")


@dataclass(frozen=True)
class BuddyAdvice:
    expression: str
    title: str
    message: str
    trigger_reason: str
    priority: int


ADVICE_CATALOG: Dict[str, List[BuddyAdvice]] = {
    "burnout": [
        BuddyAdvice(
            expression="jalan",
            title="Waktunya Jeda dari Layar",
            message="Skor kelelahanmu bertahan tinggi selama 15 menit terakhir. Berdiri dan jalan kaki sebentar selama 5 menit.",
            trigger_reason="Rata-rata skor melewati batas burnout",
            priority=5,
        ),
        BuddyAdvice(
            expression="istirahat",
            title="Istirahatkan Pikiran Sebentar",
            message="Kamu sudah bekerja dengan ketegangan tinggi cukup lama. Tinggalkan meja sejenak dan tarik napas panjang.",
            trigger_reason="Rata-rata skor melewati batas burnout",
            priority=5,
        ),
    ],
    "kantuk": [
        BuddyAdvice(
            expression="menguap",
            title="Matamu Mulai Berat",
            message="Kelopak matamu sering tertutup atau kamu baru saja menguap. Pejamkan mata selama 60 detik atau basuh wajah.",
            trigger_reason="Deteksi kantuk atau menguap berulang",
            priority=4,
        ),
        BuddyAdvice(
            expression="istirahat",
            title="Rehat Mata 20 Detik",
            message="Tanda kantuk mulai terlihat jelas. Alihkan pandangan ke objek sejauh 6 meter selama 20 detik.",
            trigger_reason="Deteksi kantuk atau menguap berulang",
            priority=4,
        ),
    ],
    "tegang_dahi": [
        BuddyAdvice(
            expression="minum",
            title="Kendurkan Otot Wajah",
            message="Alis dan dahimu terlihat menegang. Minum air putih beberapa teguk dan rilekskan bahu serta rahangmu.",
            trigger_reason="Ketegangan dahi atau tangan menyentuh kepala",
            priority=3,
        ),
        BuddyAdvice(
            expression="istirahat",
            title="Tarik Napas Perlahan",
            message="Kamu tampak sedang berkonsentrasi penuh sampai menahan ketegangan di wajah. Ambil napas dalam tiga kali.",
            trigger_reason="Ketegangan dahi atau tangan menyentuh kepala",
            priority=3,
        ),
    ],
    "postur_bungkuk": [
        BuddyAdvice(
            expression="olahraga",
            title="Luruskan Kembali Punggungmu",
            message="Posisi kepalamu mulai condong terlalu dekat ke layar. Tegakkan punggung dan putar bahu perlahan.",
            trigger_reason="Postur membungkuk terhadap baseline",
            priority=2,
        ),
    ],
    "jarang_berkedip": [
        BuddyAdvice(
            expression="minum",
            title="Kedipkan Mata dan Minum Air",
            message="Frekuensi kedipanmu menurun karena menatap layar terus-menerus. Kedipkan mata beberapa kali dan minum air.",
            trigger_reason="Frekuensi kedipan rendah",
            priority=1,
        ),
    ],
}


def pick_advice_for_reason(reason: str) -> BuddyAdvice:
    candidates = ADVICE_CATALOG.get(reason) or ADVICE_CATALOG["jarang_berkedip"]
    return random.choice(candidates)


def pick_preview_advice(expression: str) -> BuddyAdvice:
    for items in ADVICE_CATALOG.values():
        for item in items:
            if item.expression == expression:
                return item
    return BuddyAdvice(
        expression=expression if expression in VALID_EXPRESSIONS else "istirahat",
        title="Saran Istirahat Singkat",
        message="Luangkan waktu satu menit untuk merilekskan mata dan bahu sebelum melanjutkan pekerjaan.",
        trigger_reason="Pratinjau manual",
        priority=1,
    )
