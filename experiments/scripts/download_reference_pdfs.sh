#!/bin/bash
# Download PDFs for entries cited in paper/references.bib
# Output: paper/references/pdfs/<bibkey>.pdf
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/paper/references/pdfs"
mkdir -p "$OUT"

UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

dl() {
  local key="$1" url="$2"
  local dest="$OUT/${key}.pdf"
  if [ -s "$dest" ] && head -c 4 "$dest" | grep -q '%PDF'; then
    echo "SKIP $key (already present)"
    return 0
  fi
  if curl -L --fail --silent --max-time 120 -A "$UA" -o "$dest" "$url"; then
    if head -c 4 "$dest" | grep -q '%PDF'; then
      echo "OK   $key"
    else
      echo "FAIL $key (not a PDF: $url)"
      rm -f "$dest"
    fi
  else
    echo "FAIL $key ($url)"
    rm -f "$dest"
  fi
}

# ── arXiv (43) ──────────────────────────────────────────────
dl deyoung2020eraser      https://arxiv.org/pdf/1911.03429
dl thorne2018fever        https://arxiv.org/pdf/1803.05355
dl lei2016rationalizing   https://arxiv.org/pdf/1606.04155
dl camburu2018esnli       https://arxiv.org/pdf/1812.01194
dl rajani2019cose         https://arxiv.org/pdf/1903.02671
dl hendrycks2021cuad      https://arxiv.org/pdf/2103.06268
dl mathew2021hatexplain   https://arxiv.org/pdf/2012.10289
dl pavlopoulos2021toxic   https://arxiv.org/pdf/2104.08728
dl atanasova2020diagnostic https://arxiv.org/pdf/2010.12331
dl wiegreffe2022reframing https://arxiv.org/pdf/2205.09438
dl jain2019attention      https://arxiv.org/pdf/1902.10186
dl serrano2019attention   https://arxiv.org/pdf/1906.03731
dl wiegreffe2019attention https://arxiv.org/pdf/1908.04626
dl jacovi2020faithfully   https://arxiv.org/pdf/2005.00163
dl pruthi2020learning     https://arxiv.org/pdf/2005.05412
dl bastings2020elephant   https://arxiv.org/pdf/2010.13525
dl abnar2020quantifying   https://arxiv.org/pdf/2006.05695
dl ilse2018abmil          https://arxiv.org/pdf/1802.04712
dl carbonneau2018mil      https://arxiv.org/pdf/1612.03365
dl yang2016hierarchical   https://arxiv.org/pdf/1605.09623
dl shao2021transmil       https://arxiv.org/pdf/2106.00978
dl li2021dsmil            https://arxiv.org/pdf/2011.08939
dl zhou2016cam            https://arxiv.org/pdf/1512.04150
dl oquab2015object        https://arxiv.org/pdf/1412.6856
dl ribeiro2016lime        https://arxiv.org/pdf/1602.04938
dl lundberg2017shap       https://arxiv.org/pdf/1705.07874
dl alvarez2018senn        https://arxiv.org/pdf/1801.01253
dl murdoch2019definitions https://arxiv.org/pdf/1901.05315
dl doshi2017towards       https://arxiv.org/pdf/1703.04730
dl lipton2018mythos       https://arxiv.org/pdf/1606.03490
dl selvaraju2017gradcam   https://arxiv.org/pdf/1610.02391
dl belinkov2019analysis   https://arxiv.org/pdf/1904.06028
dl ratner2017snorkel      https://arxiv.org/pdf/1711.10160
dl bach2017dataprogramming https://arxiv.org/pdf/1605.07723
dl zhang2021wrench        https://arxiv.org/pdf/2004.06025
dl bach2012optimization   https://arxiv.org/pdf/1208.6116
dl jang2017gumbel         https://arxiv.org/pdf/1611.01144
dl correia2019adaptively  https://arxiv.org/pdf/1907.05572
dl devlin2019bert         https://arxiv.org/pdf/1810.04805
dl kim2014cnn             https://arxiv.org/pdf/1408.5882
dl beltagy2020longformer  https://arxiv.org/pdf/2004.05150
dl child2019sparse        https://arxiv.org/pdf/1904.10509
dl sanh2019distilbert     https://arxiv.org/pdf/1910.01108

# ── Open-access venues / author pages ──────────────────────
dl lin2021phishpedia      https://www.usenix.org/system/files/sec21-lin.pdf
dl gdpr2016               "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679&from=EN"
dl euaiact2024            "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689&from=EN"
dl zhou2004mil            https://www.lamda.nju.edu.cn/publication/techrep04.pdf
dl lu2021histopathology   https://www.biorxiv.org/content/10.1101/2021.01.12.426227v1.full.pdf
dl tibshirani1996lasso    https://statweb.stanford.edu/~tibs/lasso/lasso.pdf
dl stajano2011scam        https://www.cl.cam.ac.uk/~fms27/papers/2011-StajanoWilson-scamCACM.pdf
dl florencio2011weis      https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/WhereDoAllTheAttacksGo.pdf
dl herley2012nigerian     https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/WhyFromNigeria.pdf
dl heiding2024llm         https://ieeexplore.ieee.org/iel8/6287639/6514899/10466545.pdf
dl rudin2019stop          https://arxiv.org/pdf/1811.10154
dl klimt2004enron         https://nl.ijs.si/janes/wp-content/uploads/2014/09/klimtyang04b.pdf
dl fbi2024ic3             https://www.ic3.gov/AnnualReport/Reports/2024_IC3Report.pdf
dl verizon2024dbir        https://raw.githubusercontent.com/cybershujin/Threat-Actors-use-of-Artifical-Intelligence/main/2024-dbir-data-breach-investigations-report.pdf
dl zaidan2008explanations https://aclanthology.org/P08-2033.pdf

# ── Local copy ─────────────────────────────────────────────
if [ ! -s "$OUT/nguyen2024ai2tale.pdf" ]; then
  cp "$ROOT/paper/references/6085_AI2TALE_An_Innovative_Inf_20260215.pdf" "$OUT/nguyen2024ai2tale.pdf" && echo "OK   nguyen2024ai2tale (local copy)"
fi

echo ""
echo "=== Verification ==="
total=0; ok=0
for f in "$OUT"/*.pdf; do
  [ -e "$f" ] || continue
  total=$((total+1))
  if head -c 4 "$f" | grep -q '%PDF'; then ok=$((ok+1)); else echo "BAD: $f"; fi
done
echo "Valid PDFs: $ok / $total"
