from src.translation_utils import *
from src.dataset_utils import *

##### available datasets #####
# mgsm
# xcopa
# xstorycloze
# mkqa
# pawsx
# xnli
# xlsum

##### available languages ####
#   ace_Arab, ace_Latn, acm_Arab, acq_Arab, aeb_Arab, afr_Latn, ajp_Arab,
#   aka_Latn, amh_Ethi, apc_Arab, arb_Arab, ars_Arab, ary_Arab, arz_Arab,
#   asm_Beng, ast_Latn, awa_Deva, ayr_Latn, azb_Arab, azj_Latn, bak_Cyrl,
#   bam_Latn, ban_Latn,bel_Cyrl, bem_Latn, ben_Beng, bho_Deva, bjn_Arab, bjn_Latn,
#   bod_Tibt, bos_Latn, bug_Latn, bul_Cyrl, cat_Latn, ceb_Latn, ces_Latn,
#   cjk_Latn, ckb_Arab, crh_Latn, cym_Latn, dan_Latn, deu_Latn, dik_Latn,
#   dyu_Latn, dzo_Tibt, ell_Grek, eng_Latn, epo_Latn, est_Latn, eus_Latn,
#   ewe_Latn, fao_Latn, pes_Arab, fij_Latn, fin_Latn, fon_Latn, fra_Latn,
#   fur_Latn, fuv_Latn, gla_Latn, gle_Latn, glg_Latn, grn_Latn, guj_Gujr,
#   hat_Latn, hau_Latn, heb_Hebr, hin_Deva, hne_Deva, hrv_Latn, hun_Latn,
#   hye_Armn, ibo_Latn, ilo_Latn, ind_Latn, isl_Latn, ita_Latn, jav_Latn,
#   jpn_Jpan, kab_Latn, kac_Latn, kam_Latn, kan_Knda, kas_Arab, kas_Deva,
#   kat_Geor, knc_Arab, knc_Latn, kaz_Cyrl, kbp_Latn, kea_Latn, khm_Khmr,
#   kik_Latn, kin_Latn, kir_Cyrl, kmb_Latn, kon_Latn, kor_Hang, kmr_Latn,
#   lao_Laoo, lvs_Latn, lij_Latn, lim_Latn, lin_Latn, lit_Latn, lmo_Latn,
#   ltg_Latn, ltz_Latn, lua_Latn, lug_Latn, luo_Latn, lus_Latn, mag_Deva,
#   mai_Deva, mal_Mlym, mar_Deva, min_Latn, mkd_Cyrl, plt_Latn, mlt_Latn,
#   mni_Beng, khk_Cyrl, mos_Latn, mri_Latn, zsm_Latn, mya_Mymr, nld_Latn,
#   nno_Latn, nob_Latn, npi_Deva, nso_Latn, nus_Latn, nya_Latn, oci_Latn,
#   gaz_Latn, ory_Orya, pag_Latn, pan_Guru, pap_Latn, pol_Latn, por_Latn,
#   prs_Arab, pbt_Arab, quy_Latn, ron_Latn, run_Latn, rus_Cyrl, sag_Latn,
#   san_Deva, sat_Beng, scn_Latn, shn_Mymr, sin_Sinh, slk_Latn, slv_Latn,
#   smo_Latn, sna_Latn, snd_Arab, som_Latn, sot_Latn, spa_Latn, als_Latn,
#   srd_Latn, srp_Cyrl, ssw_Latn, sun_Latn, swe_Latn, swh_Latn, szl_Latn,
#   tam_Taml, tat_Cyrl, tel_Telu, tgk_Cyrl, tgl_Latn, tha_Thai, tir_Ethi,
#   taq_Latn, taq_Tfng, tpi_Latn, tsn_Latn, tso_Latn, tuk_Latn, tum_Latn,
#   tur_Latn, twi_Latn, tzm_Tfng, uig_Arab, ukr_Cyrl, umb_Latn, urd_Arab,
#   uzn_Latn, vec_Latn, vie_Latn, war_Latn, wol_Latn, xho_Latn, ydd_Hebr,
#   yor_Latn, yue_Hant, zho_Hans, zho_Hant, zul_Latn 

#### initial language selection ####
# langs = ["afr_Latn","arb_Arab","ban_Latn","bel_Cyrl","ben_Beng","bod_Tibt", "bos_Latn","bul_Cyrl",
# "ces_Latn", "cat_Latn","dan_Latn", "deu_Latn","eng_Latn","ell_Grek","est_Latn", 
# "fin_Latn", "fra_Latn","hat_Latn", "heb_Hebr","hin_Deva","hun_Latn", "hrv_Latn", "hye_Armn", 
# "ind_Latn", "ita_Latn","jav_Latn", "jpn_Jpan","khm_Khmr","kor_Hang", 
# "lao_Laoo","mai_Deva", "mal_Mlym", "mar_Deva", "mya_Mymr", "nno_Latn",
# "nld_Latn", "npi_Deva","pol_Latn","por_Latn", "slk_Latn","quy_Latn","ron_Latn", "rus_Cyrl", 
# "slv_Latn", "spa_Latn", "srp_Cyrl", "swe_Latn", "swh_Latn", "tam_Taml", "tel_Telu", 
# "tgl_Latn", 'tha_Thai',"tur_Latn","ukr_Cyrl", "urd_Arab", "vie_Latn" , 'yue_Hant', "zho_Hant", "zsm_Latn","zul_Latn"]

# langs_without_mgsm = ["afr_Latn","arb_Arab","ban_Latn","bel_Cyrl","bod_Tibt", "bos_Latn","bul_Cyrl",
# "ces_Latn", "cat_Latn","dan_Latn", "ell_Grek","est_Latn", 
# "fin_Latn", "hat_Latn", "heb_Hebr","hin_Deva","hun_Latn", "hrv_Latn", "hye_Armn", 
# "ind_Latn", "ita_Latn","jav_Latn","khm_Khmr","kor_Hang", 
# "lao_Laoo","mai_Deva", "mal_Mlym", "mar_Deva", "mya_Mymr", "nno_Latn",
# "nld_Latn", "npi_Deva","pol_Latn","por_Latn", "slk_Latn","quy_Latn","ron_Latn", 
# "slv_Latn", "srp_Cyrl", "swe_Latn", "tam_Taml", 
# "tgl_Latn","tur_Latn","ukr_Cyrl", "urd_Arab", "vie_Latn" , 'yue_Hant', "zsm_Latn","zul_Latn"]

# for lang in langs_without_mgsm:
#     translate_dataset(get_dataset("mgsm","en"),"mgsm",lang,model,tokenizer)

# langs_without_xcopa = ["afr_Latn","arb_Arab","ban_Latn","bel_Cyrl","ben_Beng","bod_Tibt", "bos_Latn","bul_Cyrl",
# "ces_Latn", "cat_Latn","dan_Latn", "deu_Latn","ell_Grek", 
# "fin_Latn", "fra_Latn","heb_Hebr","hin_Deva","hun_Latn", "hrv_Latn", "hye_Armn", 
# "jav_Latn", "jpn_Jpan","khm_Khmr","kor_Hang", 
# "lao_Laoo", "mai_Deva", "mal_Mlym", "mar_Deva", "mya_Mymr", "nno_Latn",
# "nld_Latn", "npi_Deva","pol_Latn","por_Latn", "slk_Latn","ron_Latn", "rus_Cyrl", 
# "slv_Latn", "spa_Latn", "srp_Cyrl", "swe_Latn", "tel_Telu", 
# "tgl_Latn", "ukr_Cyrl", "urd_Arab", 'yue_Hant', "zsm_Latn","zul_Latn"]

# download model
model_name = "facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name,token=True,src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,token=True)

langs = ["afr_Latn","arb_Arab","ban_Latn","bel_Cyrl","ben_Beng","bod_Tibt", "bos_Latn","bul_Cyrl",
"ces_Latn", "cat_Latn","dan_Latn", "deu_Latn","ell_Grek","est_Latn", 
"fin_Latn", "fra_Latn","hat_Latn", "heb_Hebr","hin_Deva","hun_Latn", "hrv_Latn", "hye_Armn", 
"ind_Latn", "ita_Latn","jav_Latn", "jpn_Jpan","khm_Khmr","kor_Hang", 
"lao_Laoo","mai_Deva", "mal_Mlym", "mar_Deva", "mya_Mymr", "nno_Latn",
"nld_Latn", "npi_Deva","pol_Latn","por_Latn", "slk_Latn","quy_Latn","ron_Latn", "rus_Cyrl", 
"slv_Latn", "spa_Latn", "srp_Cyrl", "swe_Latn", "swh_Latn", "tam_Taml", "tel_Telu", 
"tgl_Latn", 'tha_Thai',"tur_Latn","ukr_Cyrl", "urd_Arab", "vie_Latn" , 'yue_Hant', "zho_Hant", "zsm_Latn","zul_Latn"]

for lang in langs:
    translate_dataset(get_dataset_df('coinflip','eng_Latn'),"coinflip",lang,model,tokenizer)

# for lang in langs[:3]:
#     translate_dataset(get_dataset_df('shuffled_objects','eng_Latn'),"shuffled_objects",lang,model,tokenizer)
