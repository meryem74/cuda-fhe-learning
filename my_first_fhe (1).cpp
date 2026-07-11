// my_first_fhe.cpp
// Ilk elle yazilmis FHE programim: CKKS ile temel islemler,
// sure olcumu ve level (multiplicative depth) takibi.

#include "openfhe.h"
#include <chrono>
#include <cmath>

using namespace lbcrypto;

// Yardimci: bir islemin suresini milisaniye olarak olcmek icin
template <typename F>
double zamanla(F islem) {
    auto t0 = std::chrono::high_resolution_clock::now();
    islem();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    // ---------- 1) Parametreler ve CryptoContext ----------
    // Orion'da config.yml'nin yaptigini burada elle yapiyoruz.
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(2);   // carpma butcesi: 2 ardisik carpma
    parameters.SetScalingModSize(50);       // Orion'daki LogScale'in karsiligi (delta = 2^50)
    parameters.SetBatchSize(8);             // 8 slot kullanacagiz (batching/SIMD)

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);         // sifreleme/cozme
    cc->Enable(KEYSWITCH);   // carpma ve rotasyon icin gerekli
    cc->Enable(LEVELEDSHE);  // seviyeli homomorfik islemler

    std::cout << "Ring dimension (N): " << cc->GetRingDimension() << std::endl;

    // ---------- 2) Anahtar uretimi ----------
    auto keys = cc->KeyGen();                          // public + secret key
    cc->EvalMultKeyGen(keys.secretKey);                // relinearization key (carpma icin)
    cc->EvalRotateKeyGen(keys.secretKey, {1, 2, -1});  // her rotasyon miktari icin ayri anahtar!

    // ---------- 3) Encode + Encrypt ----------
    // cleartext -> plaintext -> ciphertext zinciri (Orion makalesi Section 2)
    std::vector<double> x1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    std::vector<double> x2 = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    // ---------- 4) Uc temel islem + sure olcumu ----------
    Ciphertext<DCRTPoly> cAdd, cMult, cRot;

    double tAdd  = zamanla([&]() { cAdd  = cc->EvalAdd(c1, c2);  });
    double tMult = zamanla([&]() { cMult = cc->EvalMult(c1, c2); });
    double tRot  = zamanla([&]() { cRot  = cc->EvalRotate(c1, 1); });

    std::cout << "\n--- Islem sureleri ---" << std::endl;
    std::cout << "EvalAdd   : " << tAdd  << " ms" << std::endl;
    std::cout << "EvalMult  : " << tMult << " ms  (key-switching + rescaling icerir)" << std::endl;
    std::cout << "EvalRotate: " << tRot  << " ms  (key-switching icerir)" << std::endl;

    // ---------- 5) Level takibi: multiplicative depth'i canli gor ----------
    // GetLevel() = su ana kadar TUKETILEN seviye sayisi.
    std::cout << "\n--- Level takibi (depth = 2) ---" << std::endl;
    std::cout << "c1 (taze)          -> tuketilen level: " << c1->GetLevel() << std::endl;
    std::cout << "c1*c2 (1 carpma)   -> tuketilen level: " << cMult->GetLevel() << std::endl;

    auto cMult2 = cc->EvalMult(cMult, c1);  // 2. carpma: butcenin sonu
    std::cout << "(c1*c2)*c1 (2 carpma) -> tuketilen level: " << cMult2->GetLevel() << std::endl;
    // 3. bir carpma denersek program hata verir; cunku depth=2 ile
    // parametre urettik. Denemek istersen alttaki satiri ac:
    // auto cMult3 = cc->EvalMult(cMult2, c1);

    // ---------- 6) Decrypt + dogruluk kontrolu ----------
    Plaintext rAdd, rMult, rRot;
    cc->Decrypt(keys.secretKey, cAdd,  &rAdd);
    cc->Decrypt(keys.secretKey, cMult, &rMult);
    cc->Decrypt(keys.secretKey, cRot,  &rRot);
    rAdd->SetLength(8); rMult->SetLength(8); rRot->SetLength(8);

    std::cout << "\n--- Sonuclar (tum slotlara ayni anda uygulandi = batching) ---" << std::endl;
    std::cout << "x1 + x2  : " << rAdd;
    std::cout << "x1 * x2  : " << rMult;
    std::cout << "rot(x1,1): " << rRot;

    // CKKS yaklasik bir sema: beklenen degerle arasindaki maks hatayi olc
    auto vals = rMult->GetRealPackedValue();
    double maxErr = 0.0;
    for (size_t i = 0; i < 8; i++)
        maxErr = std::max(maxErr, std::fabs(vals[i] - x1[i] * 0.5));
    std::cout << "\nCarpmada maksimum hata: " << maxErr
              << "  (~" << -std::log2(maxErr) << " bit hassasiyet)" << std::endl;

    return 0;
}
