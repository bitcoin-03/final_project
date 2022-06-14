#%% Import library
import numpy as np
import pandas as pd
import matplotlib as mpl
import os
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
from scipy.stats import shapiro
from torch import save

#%% Custom functions
def save_fig(path:str=None, dpi:int=150):
    plt.savefig(path+'.png', dpi=dpi)
    plt.close()


def user_actor_gender_class(original_gender: str, actor_gender: str):
    if original_gender=='여성' and actor_gender=='여성':
        return '여성-여성'
    elif original_gender=='여성' and actor_gender=='남성':
        return '여성-남성'
    elif original_gender=='남성' and actor_gender=='여성':
        return '남성-여성'
    elif original_gender=='남성' and actor_gender=='남성':
        return '남성-남성'
    else:
        print('Input combination is wrong!')
        raise NotImplementedError


def transform_beautyGAN_feedback_into_int(original:str):
    if original=='배우와 더 닮아보인다.':
        return 1
    elif original=='차이가 없다.':
        return 0
    elif original=='더 이상하다.':
        return -1
    else:
        print(f'Wrong in the original feedback of beautyGAN {original}')
        raise NotImplementedError

#%% Set matplotlib style
# Set total style
# plt.style.use('seaborn')

# Korean font
font_files = font_manager.findSystemFonts(fontpaths='./fonts')

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

kfont_path = "./fonts/NanumBarunGothic.ttf"
kfont_nm = font_manager.FontProperties(fname=kfont_path).get_name()
rc('font', family=kfont_nm)

mpl.rcParams['axes.unicode_minus'] = False

# Font size
SMALL_SIZE=10
MEDIUM_SIZE=15
LARGE_SIZE=20

plt.rc('font', size=MEDIUM_SIZE, weight='bold')
plt.rc('xtick', labelsize=LARGE_SIZE)
plt.rc('ytick', labelsize=LARGE_SIZE)
plt.rc('axes', labelsize=LARGE_SIZE, labelweight='bold')
plt.rc('legend', fontsize=LARGE_SIZE)

#%% Make figure save directory
fig_dir_nm = 'figures_feedback_analysis/'
if not os.path.isdir(fig_dir_nm):
    os.mkdir(fig_dir_nm)

#%% Load feedback files and preprocess
# Load feedback file
file_nm = '배우고 싶니_ 서비스 피드백 설문지(응답) - 설문지 응답 220613 1428.csv'
df_feedback_raw = pd.read_csv(file_nm)
file_date = file_nm[-15:-4]
print(file_date)

# Remove unnecessary rows
df_feedback = df_feedback_raw[~df_feedback_raw['타임스탬프'].isnull()]

# Make a pivot table
# pivot_tbl = pd.pivot_table(
#     df_feedback,
#     index='1.  본인의 성별',
#     columns='개발 경험이 있으신가요?',
#     values=,
#     aggfunc=['mean', 'median', 'sum'],
# )

#%% Add additional columns for analysis
df_feedback['성별: 사용자-닮은 배우'] = df_feedback.apply(
    lambda x: user_actor_gender_class(x['1.  본인의 성별'], x['2. 결과 연예인의 성별']), axis=1)
df_feedback['beautyGAN 만족도'] = df_feedback.apply(
    lambda x: transform_beautyGAN_feedback_into_int(
        x['6. 닮은 배우의 화장을 입힌 결과가 화장을 입히기 전과 비교했을 때 어떠신가요?']), axis=1)

#%% Check basic correlation between features
check_normality = {col: [
    shapiro(df_feedback[col]).pvalue]
    for col in df_feedback.columns if df_feedback[col].dtype != 'object'}
check_normality_series = pd.Series(
    index=[k for k in check_normality.keys()],
    data=[v for k, v in check_normality.items()])

# plot - Shapiro p-value of each columns as a bar chart
fig_col_shapiro, ax_col_shapiro = plt.subplots(1, 1, figsize=(12, 6))

ax_col_shapiro.barh(
    check_normality_series.index, 
    check_normality_series.values
    )
ax_col_shapiro.axvline(0.05, linestyle='--', linewidth=2, color='r', label='0.05')
ax_col_shapiro.set_xscale('log')
ax_col_shapiro.legend()
ax_col_shapiro.set_xlim([1e-7, 0.15])
ax_col_shapiro.invert_yaxis()
ax_col_shapiro.set_yticklabels(
    ['서비스 만족도', '서비스 체감 속도', '배우가 닮게 느껴지는 정도', '메이크업 만족도'])

fig_col_shapiro.tight_layout(rect=[0, 0.03, 1, 0.95])
ax_col_shapiro.set_title('Shapiro test p-value')

save_fig(os.path.join(fig_dir_nm, 'Numerical column shapiro'))

# plot - spearman correlation heatmap
fig_spearman, ax_spearman = plt.subplots(1, 1, figsize=(12, 12))

sns.heatmap(
    df_feedback.corr(method='spearman'),
    ax=ax_spearman,
    annot=True,
    square=True,
    annot_kws={"size":25})
ax_spearman.set_xticklabels(['서비스 만족도', '서비스 체감 속도', '배우가 닮게 느껴지는 정도', '메이크업 만족도'])
ax_spearman.set_yticklabels(['서비스 만족도', '서비스 체감 속도', '배우가 닮게 느껴지는 정도', '메이크업 만족도'])

fig_spearman.tight_layout()

save_fig(os.path.join(fig_dir_nm, 'Spearman Heatmap'))

#%% Analyze user-actor gender matching result
df_by_gender_match = df_feedback.groupby(by='성별: 사용자-닮은 배우')
num_gender_match = df_feedback['성별: 사용자-닮은 배우'].value_counts()

# plot - user actor gender match simple count
fig_user_gender_count, ax_user_gender_count = plt.subplots(1, 1, figsize=(12, 8))

sns.countplot(df_feedback['성별: 사용자-닮은 배우'], ax=ax_user_gender_count)

save_fig(os.path.join(fig_dir_nm, '사용자-배우 성별 결과 bar chart'))

# plot - Numerical feature average by user-actor gender match
fig_mean_by_gender_match, ax_mean_by_gender_match = plt.subplots(1, 1, figsize=(12, 8))

ax_mean_by_gender_match.spines['top'].set_linewidth(0)
ax_mean_by_gender_match.spines['right'].set_linewidth(0)

ax_mean_by_gender_match.barh(
    np.array([1, 2, 3])-0.2, df_by_gender_match.median().loc['여성-여성', :],
    0.15,  
    label=f"사용자-배우 : 여성-여성 (N={num_gender_match['여성-여성']})"
    )
ax_mean_by_gender_match.barh(
    np.array([1, 2, 3]), df_by_gender_match.mean().loc['남성-여성', :],
    0.15,  
    label=f"사용자-배우 : 남성-여성 (N={num_gender_match['남성-여성']})"
    )
ax_mean_by_gender_match.barh(
    np.array([1, 2, 3])+0.2, df_by_gender_match.mean().loc['남성-남성', :],
    0.15,  
    label=f"사용자-배우 : 남성-남성 (N={num_gender_match['남성-남성']})"
    )

ax_mean_by_gender_match.set_yticks([1, 2, 3])
ax_mean_by_gender_match.set_yticklabels(['여성-여성', '남성-여성', '여성-여성'])
ax_mean_by_gender_match.set_xlabel('사용자 긍정 평가 (0 최하, 5 최상)')
ax_mean_by_gender_match.legend()
ax_mean_by_gender_match.set_xlim([0, 5.5])
ax_mean_by_gender_match.set_ylim([0, 5])
ax_mean_by_gender_match.grid(axis='x')

fig_mean_by_gender_match.suptitle('수치형 자료 중위수 비교 (사용자-배우 성별 매칭 결과 기준)'+file_date)

save_fig(os.path.join(fig_dir_nm, '수치형 자료 중위수 비교 (사용자-배우 성별 매칭 결과 기준)'))


#%% Analyze feedback by coding experiences
df_by_coding = df_feedback.groupby(by='개발 경험이 있으신가요?')

num_code_positive, num_code_negative = \
    df_feedback['개발 경험이 있으신가요?'].value_counts()['네'],\
    df_feedback['개발 경험이 있으신가요?'].value_counts()['아니오']

# plot - Numerical feature average by coding experiences
fig_mean_by_code, ax_mean_by_code = plt.subplots(1, 1, figsize=(12, 8))

ax_mean_by_code.spines['top'].set_linewidth(0)
ax_mean_by_code.spines['right'].set_linewidth(0)

ax_mean_by_code.barh(
    np.array([1, 2, 3])-0.15, df_by_coding.mean().loc['네', :],
    0.25,  
    label=f'코딩경험 있음 (N={num_code_positive})'
    )
ax_mean_by_code.barh(
    np.array([1, 2, 3])+0.15, df_by_coding.mean().loc['아니오', :], 
    0.25, 
    label=f'코딩경험 없음 (N={num_code_negative})'
    )
ax_mean_by_code.invert_yaxis()

ax_mean_by_code.set_yticks([1, 2, 3])
ax_mean_by_code.set_yticklabels(['서비스 만족도', '서비스 체감 속도', '배우가 닮게 느껴지는 정도'])
ax_mean_by_code.set_xlabel('사용자 긍정 평가 (0 최하, 5 최상)')
ax_mean_by_code.legend()
ax_mean_by_code.set_xlim([0, 5])
ax_mean_by_code.set_ylim([0, 4.5])
fig_mean_by_code.tight_layout()

fig_mean_by_code.suptitle('수치형 자료 평균 비교 (코딩 경험 유무 기준)'+file_date)

save_fig(os.path.join(fig_dir_nm, '수치형 자료 평균 비교 (코딩 경험 유무 기준)'))


# plot - Numerical feature average by coding experiences
fig_median_by_code, ax_median_by_code = plt.subplots(1, 1, figsize=(12, 8))

ax_median_by_code.spines['top'].set_linewidth(0)
ax_median_by_code.spines['right'].set_linewidth(0)

ax_median_by_code.barh(
    np.array([1, 2, 3])-0.15, df_by_coding.median().loc['네', :],
    0.25,  
    label=f'코딩경험 있음 (N={num_code_positive})'
    )
ax_median_by_code.barh(
    np.array([1, 2, 3])+0.15, df_by_coding.median().loc['아니오', :], 
    0.25, 
    label=f'코딩경험 없음 (N={num_code_negative})'
    )
ax_median_by_code.invert_yaxis()

ax_median_by_code.set_yticks([1, 2, 3])
ax_median_by_code.set_yticklabels(['서비스 만족도', '서비스 체감 속도', '배우가 닮게 느껴지는 정도'])
ax_median_by_code.set_xlabel('사용자 긍정 평가 (0 최하, 5 최상)')
ax_median_by_code.legend()
ax_median_by_code.set_xlim([0, 5])
ax_median_by_code.set_ylim([0, 4.5])
fig_median_by_code.tight_layout()

fig_median_by_code.suptitle('수치형 자료 중앙값 비교 (코딩 경험 유무 기준)'+file_date)

save_fig(os.path.join(fig_dir_nm, '수치형 자료 중앙값 비교 (코딩 경험 유무 기준)'))

#%% Analyze feedback by gender
df_by_gender = df_feedback.groupby(by='1.  본인의 성별')

num_female, num_male = \
    df_feedback['1.  본인의 성별'].value_counts()['여성'],\
    df_feedback['1.  본인의 성별'].value_counts()['남성']

# plot - Numerical feature average by gender
fig_mean_by_gender, ax_mean_by_gender = plt.subplots(1, 1, figsize=(12, 8))

ax_mean_by_gender.spines['top'].set_linewidth(0)
ax_mean_by_gender.spines['right'].set_linewidth(0)

ax_mean_by_gender.barh(
    np.array([1, 2, 3])-0.15, df_by_gender.mean().loc['여성', :],
    0.25,  
    label=f'여성 (N={num_female})'
    )
ax_mean_by_gender.barh(
    np.array([1, 2, 3])+0.15, df_by_gender.mean().loc['남성', :], 
    0.25, 
    label=f'남성 (N={num_male})'
    )
ax_mean_by_gender.invert_yaxis()

ax_mean_by_gender.set_yticks([1, 2, 3])
ax_mean_by_gender.set_yticklabels(['서비스 만족도', '서비스 체감 속도', '배우가 닮게 느껴지는 정도'])
ax_mean_by_gender.set_xlabel('사용자 긍정 평가 (0 최하, 5 최상)')
ax_mean_by_gender.legend()
ax_mean_by_gender.set_xlim([0, 5])
ax_mean_by_gender.set_ylim([0, 4.5])
fig_mean_by_gender.tight_layout()

fig_mean_by_gender.suptitle('수치형 자료 평균 비교 (성별 기준)'+file_date)

save_fig(os.path.join(fig_dir_nm, '수치형 자료 평균 비교 (성별 기준)'))

# plot - Numerical feature median by gender
fig_median_by_gender, ax_median_by_gender = plt.subplots(1, 1, figsize=(12, 8))

ax_median_by_gender.spines['top'].set_linewidth(0)
ax_median_by_gender.spines['right'].set_linewidth(0)

ax_median_by_gender.barh(
    np.array([1, 2, 3])-0.15, df_by_gender.median().loc['여성', :],
    0.25,  
    label=f'여성 (N={num_female})'
    )
ax_median_by_gender.barh(
    np.array([1, 2, 3])+0.15, df_by_gender.median().loc['남성', :], 
    0.25, 
    label=f'남성 (N={num_male})'
    )
ax_median_by_gender.invert_yaxis()

ax_median_by_gender.set_yticks([1, 2, 3])
ax_median_by_gender.set_yticklabels(['서비스 만족도', '서비스 체감 속도', '배우가 닮게 느껴지는 정도'])
ax_median_by_gender.set_xlabel('사용자 긍정 평가 (0 최하, 5 최상)')
ax_median_by_gender.legend()
ax_median_by_gender.set_xlim([0, 5])
ax_median_by_gender.set_ylim([0, 4.5])
fig_median_by_gender.tight_layout()

fig_median_by_gender.suptitle('수치형 자료 중앙값 비교 (성별 기준)'+file_date)

save_fig(os.path.join(fig_dir_nm, '수치형 자료 중앙값 비교 (성별 기준)'))

#%% Analysis by both gender and coding experience
# plot - satisfaction bar chart by gender and coding experience
fig_mean_by_two_factor = sns.catplot(
    x='6. 닮은 배우의 화장을 입힌 결과가 화장을 입히기 전과 비교했을 때 어떠신가요?',#'3. 전반적인 만족도', 
    # y='1.  본인의 성별',
    col='1.  본인의 성별',
    hue='개발 경험이 있으신가요?',
    data=df_feedback,
    kind='count',
    height=12,
    aspect=1.2,
    ci=95,
    orient='h',
    estimator=np.mean,
    )
    
# fig_mean_by_two_factor.ax.set_ylabel('성별')
# fig_mean_by_two_factor.ax.set_xlabel('사용자 긍정 평가 (1: 최하, 5: 최상)')
# fig_mean_by_two_factor.ax.set_xlim([0, 5])
fig_mean_by_two_factor.tight_layout()

fig_mean_by_two_factor.fig.suptitle('성별, 코딩 경험 유무에 따른 사용자 긍정 평가 평균'+file_date)

save_fig(os.path.join(fig_dir_nm, '성별, 코딩 경험 유무에 따른 사용자 긍정 평가 평균'))