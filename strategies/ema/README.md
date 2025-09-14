# EMA (Exponential Moving Average) Çapraz Stratejisi

## Strateji Nedir?

İki farklı periyotlu üssel hareketli ortalamanın kesişimlerini kullanarak trend değişimlerini tespit eden klasik bir trend takip stratejisidir.

## Nasıl Çalışır?

- **Hızlı EMA, Yavaş EMA'yı yukarı keser**: **ALIM sinyali** (Golden Cross)
- **Hızlı EMA, Yavaş EMA'yı aşağı keser**: **SATIM sinyali** (Death Cross)
- **Kesişim yok**: **BEKLE**

## Temel Mantık

Kısa vadeli EMA (örn: 12 gün), uzun vadeli EMA'dan (örn: 26 gün) daha hızlı tepki verir. Hızlı EMA'nın yavaş EMA'yı yukarı kesmesi yükseliş trendinin, aşağı kesmesi ise düşüş trendinin başlangıcını işaret eder.