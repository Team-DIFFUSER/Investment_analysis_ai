from .stocks.lg_electronics import LGElectronicsModel
from .stocks.samsung_electronics import SamsungElectronicsModel
from .stocks.sk_hynix import SKHynixModel
from .stocks.samsung_biologics import SamsungBiologicsModel
from .stocks.lg_chemical import LGEnergySolutionModel
from .stocks.hanwha import HanwhaAerospaceModel
from .stocks.hyundai_motor import HyundaiMotorModel
from .stocks.kia import KiaModel
from .stocks.hd_hyundai import HDHyundaiModel
from .stocks.lg_energy_solution import LGEnergySolutionModel
from .stocks.samsung_electronics_preferred import SamsungElectronicsPreferredModel
from .stocks.kb_financial_group import KBFinancialGroupModel
from .stocks.doosan_enerbility import DoosanEnerbilityModel
from .stocks.celltrion import CelltrionModel
from .stocks.naver import NaverModel
from .stocks.hanwha_ocean import HanwhaOceanModel
from .stocks.shinhan_financial_group import ShinhanFinancialGroupModel
from .stocks.hyundai_mobis import HyundaiMobisModel
from .stocks.hd_korea_shipbuilding_offshore import HDKoreaShipbuildingOffshoreModel
from .stocks.samsung_life_insurance import SamsungLifeInsuranceModel
from .stocks.posco_holdings import PoscoHoldingsModel
from .stocks.korea_electric_power import KoreaElectricPowerModel
from .stocks.hana_financial_group import HanaFinancialGroupModel
from .stocks.hmm import HMMModel
from .stocks.hyundai_rotem import HyundaiRotemModel
from .stocks.meritz_financial_group import MeritzFinancialGroupModel
from .stocks.samsung_fire_marine import SamsungFireMarineModel
from .stocks.sk_square import SKSquareModel
from .stocks.woori_financial_group import WooriFinancialGroupModel
from .stocks.hd_hyundai_electric import HDHyundaiElectricModel
from .stocks.samsung_heavy_industries import SamsungHeavyIndustriesModel
from .stocks.sk_innovation import SKInnovationModel
from .stocks.samsung_sdi import SamsungSDIModel
from .stocks.korea_zinc import KoreaZincModel
from .stocks.kt_g import KTGModel
from .stocks.krafton import KraftonModel
from .stocks.industrial_bank_of_korea import IndustrialBankOfKoreaModel
from .stocks.sk import SKModel
from .stocks.kt import KTModel
from .stocks.lig_nex1 import LIGNex1Model
from .stocks.kakao_bank import KakaoBankModel

ALL_STOCK_MODELS = {
    'LG전자': LGElectronicsModel,
    '삼성전자': SamsungElectronicsModel,
    'SK하이닉스': SKHynixModel,
    '삼성바이오로직스': SamsungBiologicsModel,
    'LG화학': LGEnergySolutionModel,
    '한화': HanwhaAerospaceModel,
    '현대차': HyundaiMotorModel,
    '기아': KiaModel,
    'HD현대': HDHyundaiModel,
    'LG에너지솔루션': LGEnergySolutionModel,
    '한화에어로스페이스': HanwhaAerospaceModel,
    '삼성전자우': SamsungElectronicsPreferredModel,
    'KB금융': KBFinancialGroupModel,
    '두산에너빌리티': DoosanEnerbilityModel,
    '셀트리온': CelltrionModel,
    'NAVER': NaverModel,
    '한화오션': HanwhaOceanModel,
    '신한지주': ShinhanFinancialGroupModel,
    '현대모비스': HyundaiMobisModel,
    'HD한국조선해양': HDKoreaShipbuildingOffshoreModel,
    '삼성생명': SamsungLifeInsuranceModel,
    'POSCO홀딩스': PoscoHoldingsModel,
    '한국전력': KoreaElectricPowerModel,
    '하나금융지주': HanaFinancialGroupModel,
    'HMM': HMMModel,
    '현대로템': HyundaiRotemModel,
    '메리츠금융지주': MeritzFinancialGroupModel,
    '삼성화재': SamsungFireMarineModel,
    'SK스퀘어': SKSquareModel,
    '우리금융지주': WooriFinancialGroupModel,
    'HD현대일렉트릭': HDHyundaiElectricModel,
    '삼성중공업': SamsungHeavyIndustriesModel,
    'SK이노베이션': SKInnovationModel,
    '삼성SDI': SamsungSDIModel,
    '고려아연': KoreaZincModel,
    'KT&G': KTGModel,
    '크래프톤': KraftonModel,
    '기업은행': IndustrialBankOfKoreaModel,
    'SK': SKModel,
    'KT': KTModel,
    'LIG넥스원': LIGNex1Model,
    '카카오뱅크': KakaoBankModel,
}