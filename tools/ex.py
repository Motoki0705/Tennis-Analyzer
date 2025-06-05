import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from pathlib import Path
from matplotlib.ticker import MultipleLocator  # 追加

# ------------------------------------------------------------------------------
# 日本語文字化け防止のためのフォント指定（環境に合わせて変更してください）
# ------------------------------------------------------------------------------
mpl.rcParams['font.family'] = 'MS Gothic'
mpl.rcParams['axes.unicode_minus'] = False  # マイナス記号が化けないように

# -----------------------------------------------
# 1. 東京都人口・GDP・移住相談件数データをハードコード
# -----------------------------------------------
years = list(range(1957, 2025))
population = [
    8259474, 8524433, 8828705, 9106349, 9398171, 9665590, 9932080, 10197370,
    10385484, 10616733, 10719948, 10842612, 10997792, 11109453, 11199482, 11294572,
    11360670, 11389985, 11392619, 11415480, 11422580, 11426018, 11430896, 11422630,
    11415037, 11431882, 11479125, 11543806, 11862356, 11682315, 11740396, 11740361,
    11718720, 11698060, 11684927, 11683316, 11666227, 11627577, 11598634, 11587726,
    11602642, 11641308, 11694934, 11750351, 11823029, 11907350, 11996211, 12074598,
    12161029, 12247024, 12339259, 12433235, 12517299, 12591643, 12646745, 12686067,
    12740088, 12807631, 12880144, 12966307, 13043707, 13115848, 13189049, 13257596,
    13297089, 13260553, 13260553, 14105098
]
gdp = [
    None, None, None, None, None, None, None, None, None, None,  # 1957-1966
    None, None, None, None, None, None, None, None, None, None,  # 1967-1976
    None, None, None,                                        # 1977-1979
    254_527, 273_633, 287_278, 298_779, 317_557, 341_064,     # 1980-1985
    356_950, 371_216, 399_086, 429_661, 464_092, 492_015,     # 1986-1991
    503_922, 506_992, 511_959, 525_300, 538_660, 542_508,     # 1992-1997
    534_564, 530_299, 537_614, 527_411, 523_466, 526_270,     # 1998-2003
    529_638, 534_106, 537_258, 538_486, 516_175, 497_364,     # 2004-2009
    504_874, 500_046, 499_421, 512_678, 523_423, 540_741,     # 2010-2015
    544_830, 555_713, 556_571, 563_845, 539_009, 553_673,     # 2016-2021
    566_770, 597_059, 615_568                              # 2022-2024
]

consultations = {
    2008: 2475, 2009: 3823, 2010: 6021, 2011: 7062, 2012: 6445,
    2013: 9653, 2014: 12430, 2015: 21584, 2016: 26426, 2017: 33165,
    2018: np.nan, 2019: np.nan, 2020: np.nan, 2021: np.nan,
    2022: np.nan, 2023: np.nan, 2024: np.nan
}

# -----------------------------------------------
# 2. DataFrame を作成し、増加率を計算
# -----------------------------------------------
df = pd.DataFrame({
    'Year': years,
    'Population': population,
    'GDP': gdp
})
df['Consultations'] = df['Year'].map(consultations)
df['Population_growth'] = df['Population'].pct_change() * 100
df['GDP_growth'] = df['GDP'].pct_change() * 100

# 単位変換
df['Population_thousands'] = df['Population'] / 1000
df['Consultations_tens'] = df['Consultations'] / 10

# -----------------------------------------------
# 3. グラフを作成し、画像ファイルとして保存
# -----------------------------------------------
mask_pop_gdp = (df['Year'] >= 1957) & (df['Year'] <= 2024)
mask_cons = (df['Year'] >= 2008) & (df['Year'] <= 2024)

years_all = df.loc[mask_pop_gdp, 'Year']
pop_thousands = df.loc[mask_pop_gdp, 'Population_thousands']
pop_growth = df.loc[mask_pop_gdp, 'Population_growth']
gdp_growth = df.loc[mask_pop_gdp, 'GDP_growth']

years_cons = df.loc[mask_cons, 'Year']
cons_tens = df.loc[mask_cons, 'Consultations_tens']

fig, ax1 = plt.subplots(figsize=(12, 7))

# 左軸: 東京都人口（千人）(折れ線) & 移住相談件数（10件）(棒グラフ)
ax1.plot(
    years_all,
    pop_thousands,
    label='東京都人口（千人）',
    linewidth=2,
    marker='o',
    linestyle='-',
    color='tab:blue'
)
ax1.bar(
    years_cons,
    cons_tens,
    label='移住相談件数（10件）',
    alpha=0.6,
    width=0.6,
    color='tab:gray'
)

# ─── ここから目盛を千人（1000単位）に変更 ──────────────────────
# 「MultipleLocator(1000)」で、左縦軸（千人）の主目盛間隔を 1000 に設定
ax1.yaxis.set_major_locator(MultipleLocator(1000))
# もし必要であれば、補助目盛（minor tick）も表示したい場合は以下を追加
# from matplotlib.ticker import AutoMinorLocator
# ax1.yaxis.set_minor_locator(AutoMinorLocator(2))  # 1000 の半分＝500 ごとに補助目盛
# ───────────────────────────────────────────────────────

ax1.set_xlabel('年度（西暦）', fontsize=12)
ax1.set_ylabel('人口（千人）／移住相談件数（10件）', fontsize=12)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xlim(1956.5, 2024.5)

# 右軸: 人口増加率（％） & GDP増加率（％）
ax2 = ax1.twinx()
ax2.plot(
    years_all,
    pop_growth,
    label='人口増加率（％）',
    linewidth=2,
    marker='s',
    linestyle='--',
    color='tab:green'
)
ax2.plot(
    years_all,
    gdp_growth,
    label='GDP増加率（％）',
    linewidth=2,
    marker='^',
    linestyle='-.',
    color='tab:red'
)
ax2.set_ylabel('増加率（％）', fontsize=12)
ax2.tick_params(axis='y', labelcolor='tab:green')

# 凡例をまとめる
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc='upper left',
    fontsize=10,
    frameon=True,
    borderpad=0.5
)

plt.title(
    '1957～2024年 東京都人口・人口増加率・GDP増加率・移住相談件数',
    fontsize=14,
    pad=15
)
plt.tight_layout()

# 画像を相対パス「plot.png」で保存（スクリプトと同じフォルダに出力されます）
output_img_path = Path(__file__).parent / 'plot.png'
fig.savefig(str(output_img_path), dpi=300)
plt.close(fig)

# -----------------------------------------------
# 4. Excel ワークブックを作成し、データとグラフ画像を埋め込む
# -----------------------------------------------
excel_path = Path(__file__).parent / 'tokyo_report.xlsx'
with pd.ExcelWriter(str(excel_path), engine='xlsxwriter') as writer:
    # 4-1. "Data" シートに生データを書き込む
    df_to_excel = df.copy()
    df_to_excel = df_to_excel.loc[:, [
        'Year', 'Population', 'Population_thousands',
        'Population_growth', 'GDP', 'GDP_growth',
        'Consultations', 'Consultations_tens'
    ]]
    df_to_excel.to_excel(writer, sheet_name='Data', index=False)

    # 4-2. "Chart" シートを作成し、先ほど保存した画像を B2 セルに挿入
    workbook = writer.book
    worksheet = workbook.add_worksheet('Chart')
    worksheet.insert_image(
        'B2',
        str(output_img_path),
        {'x_scale': 0.8, 'y_scale': 0.8}
    )

print(f"✔ グラフ画像: {output_img_path}")
print(f"✔ Excel ファイル: {excel_path}")
