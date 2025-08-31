from .stocks.lg_electronics import LGElectronicsModel
from .stocks.samsung_electronics import SamsungElectronicsModel
from .stocks.sk_hynix import SKHynixModel
from .stocks.samsung_biologics import SamsungBiologicsModel
from .stocks.lg_chemical import LGChemicalModel
from .stocks.hanwha import HanwhaModel
from .stocks.hyundai_motor import HyundaiMotorModel
from .stocks.kia import KiaModel
from .stocks.hd_hyundai import HDHyundaiModel
from .stocks.naver import NaverModel
from .stocks.hyundai_mobis import HyundaiMobisModel
from .stocks.samsung_life_insurance import SamsungLifeInsuranceModel
from .stocks.hyundai_rotem import HyundaiRotemModel
from .stocks.samsung_fire_marine import SamsungFireMarineModel
from .stocks.hd_hyundai_electric import HDHyundaiElectricModel
from .stocks.samsung_heavy_industries import SamsungHeavyIndustriesModel
from .stocks.sk_innovation import SKInnovationModel
from .stocks.samsung_sdi import SamsungSDIModel
from .stocks.sk import SKModel
from .stocks.sk_telecom import SKTelecomModel
from .stocks.kakao import KakaoModel
from .stocks.kakao_bank import KakaoBankModel

ALL_STOCK_MODELS = {
    'LG전자': LGElectronicsModel,
    '삼성전자': SamsungElectronicsModel,
    'SK하이닉스': SKHynixModel,
    '삼성바이오로직스': SamsungBiologicsModel,
    'LG화학': LGChemicalModel,
    '한화': HanwhaModel,
    '현대차': HyundaiMotorModel,
    '기아': KiaModel,
    'HD현대': HDHyundaiModel,
    'NAVER': NaverModel,
    '현대모비스': HyundaiMobisModel,
    '삼성생명': SamsungLifeInsuranceModel,
    '현대로템': HyundaiRotemModel,
    '삼성화재': SamsungFireMarineModel,
    'HD현대일렉트릭': HDHyundaiElectricModel,
    '삼성중공업': SamsungHeavyIndustriesModel,
    'SK이노베이션': SKInnovationModel,
    '삼성SDI': SamsungSDIModel,
    'SK': SKModel,
    'SK텔레콤': SKTelecomModel,
    '카카오': KakaoModel,
    '카카오뱅크': KakaoBankModel,
}