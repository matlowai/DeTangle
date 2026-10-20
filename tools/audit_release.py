#!/usr/bin/env python3
"""Audit the Git release payload and an optional generated static site.

Default scope: tracked worktree files, staged Git blobs, and untracked nonignored
files. Ignored local build products are outside the payload; force-added ones are
checked. Findings contain release-relative filenames, never the private text.
"""

import argparse
import ast
from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO, StringIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tokenize

# Frozen hashes of the explicitly approved display assets. Updating this catalog
# is a reviewed content change, not something the auditor does automatically.
APPROVED_IMAGES = {
    'examples/beach-comparison-171018.jpg': '80cc8faeada98e573f4c08e23bb794e28eb44fce8cd3543a7cf8615fb4957eb9',
    'examples/creative-scenes/images/arcade-171017.jpg': '68d4c222329ec5fbcbab0b509016868a375118b2f18123c483e6939b25048eb1',
    'examples/creative-scenes/images/arcade-171018.jpg': 'c23ee537feace6ca4b1e990a1728dfd9365eb0b1da1b322d439baa442eb59cd9',
    'examples/creative-scenes/images/lighthouse-171017.jpg': '5a110435ba7646e6e6650a879b948de0cfd59eee752846c37b63ecdc0b1e76a3',
    'examples/creative-scenes/images/lighthouse-171018.jpg': '686d1857a0ac4a69e77300487316aed23baac245b7231e446397ad7f3f9b270d',
    'examples/creative-scenes/images/space_station-171017.jpg': '1a1fc7666d29eae87dfd15abbc75b280b5fb44965c7abe454ec9e1dc944d954a',
    'examples/creative-scenes/images/space_station-171018.jpg': '3239e5b1492680d23c6041d940bf27a219ff23f338646a593eafe9acce9fd74a',
    'examples/creative-scenes/images/wizard_workshop-171017.jpg': '329c9e7b2d39f16a0edafd5fd7e57233ec3df1b3e508b2595aff70da3f477c5a',
    'examples/creative-scenes/images/wizard_workshop-171018.jpg': '881f7792345beae23ab3eba3f1e83991661e1ee4234922b0446acd87e098a292',
    'lab/src/deltascope/case_studies/figures/beach-strengths.png': '9042f0193f04e6dc0528127e182a2a46054f77e45140379c9f4efc8f9d6bea96',
    'lab/src/deltascope/case_studies/images/arcade-generic-base-100-171017.jpg': '8c9f10d20ab2909d1400fff64c28ee5ae76b75c5d8e04f9192fd0663a38f16f4',
    'lab/src/deltascope/case_studies/images/arcade-generic-base-100-171018.jpg': 'c32e90dd6d41d15cf36a374c40d9d3abf603e72a96eeb0649bdb9761a5b4c4ad',
    'lab/src/deltascope/case_studies/images/arcade-generic-dop-100-171017.jpg': 'f600ec25f844545a5b31f19ecb4168d56bd006b08053f00375fc263a3982f19c',
    'lab/src/deltascope/case_studies/images/arcade-generic-dop-100-171018.jpg': '39133ad5c70669f78d139435bb65ce3ab99f7a0cf62a55452aa943f7533990f3',
    'lab/src/deltascope/case_studies/images/arcade-generic-dop-125-171017.jpg': '57afff8e6a5da03f96eb328ccd5fc93907b35360d80f6c840e8299e25c84a5d2',
    'lab/src/deltascope/case_studies/images/arcade-generic-dop-125-171018.jpg': '21fb557745e90491a5865b46c8392f3347332335fdad0f35f799200c9f8b1a68',
    'lab/src/deltascope/case_studies/images/arcade-generic-dop-150-171017.jpg': '9e63ba9f1be753aa6bf5bfc8c1b3d9181af764cbcbad79bfe3e7ef3e0795bbd5',
    'lab/src/deltascope/case_studies/images/arcade-generic-dop-150-171018.jpg': '85cacc701f610b4e983efbbebe12908d57d0249a1fb372d0c534abbf11548d77',
    'lab/src/deltascope/case_studies/images/arcade-generic-naive-100-171017.jpg': 'cb0faa698e3f0fd5d59f7f1f9899520e3dd1518e02d52d2b79bf6fd077991d27',
    'lab/src/deltascope/case_studies/images/arcade-generic-naive-100-171018.jpg': '6088cf91ca84d07520e92ca4f19e00da3f2f4689700bea1235842285b4471a5b',
    'lab/src/deltascope/case_studies/images/arcade-generic-naive-125-171017.jpg': '5761626d8408a3a4837f5094ee4981d1b905da85dd97a72c689e6a6d3398ad72',
    'lab/src/deltascope/case_studies/images/arcade-generic-naive-125-171018.jpg': '478f9a790d8a21531b937a925f68fef7a1febec208c83f1abdef3e50f082614b',
    'lab/src/deltascope/case_studies/images/arcade-generic-naive-150-171017.jpg': 'dc9fe1924dd642ce1d5c0e5e53c7a78145c0a8e92b5286a74d0bc2e8ec081f0b',
    'lab/src/deltascope/case_studies/images/arcade-generic-naive-150-171018.jpg': 'd5b666e792712dd28bd37f4dace3883e6df582142bbfdf835503152405c79f0d',
    'lab/src/deltascope/case_studies/images/arcade-generic-surgery-100-171017.jpg': '6949a58acbdc84551dc7d51f9ec0ce393c62c0402bdde4e50c04c18071cee11c',
    'lab/src/deltascope/case_studies/images/arcade-generic-surgery-100-171018.jpg': 'a99aed07dba9ceb9ca03a2500a0d11d2f00ed8f02c0a981c7e7dd0f5801d9c78',
    'lab/src/deltascope/case_studies/images/arcade-generic-surgery-125-171017.jpg': '39b834fc0bf2ebe2cc5bb6366e6e2727acf27cc01756e1451982876b377a8e4f',
    'lab/src/deltascope/case_studies/images/arcade-generic-surgery-125-171018.jpg': 'f776a624b5b5f989bfda995b99b5a0b7e8494b19308a401e0f55a32c13e0e74e',
    'lab/src/deltascope/case_studies/images/arcade-generic-surgery-150-171017.jpg': '456e60a3ddd0ec6013696d399cd50a08c9c8685a4cc16a72ae77c7667715bb9d',
    'lab/src/deltascope/case_studies/images/arcade-generic-surgery-150-171018.jpg': '5a4f9234f4623f6cf08f7bfc89b4bc5f0a2a4c4feb9f66bd7959ded27d3b1e3d',
    'lab/src/deltascope/case_studies/images/arcade-target-base-100-171017.jpg': '5440b7b9adfbf83fc3224d4ffa3e0f62c832d73ebf436f4ca24a9007fc956f89',
    'lab/src/deltascope/case_studies/images/arcade-target-base-100-171018.jpg': '6ae924111c64666ffa8f882e6f73a2cf3e93ea3a23e7380b295f806317d94bb3',
    'lab/src/deltascope/case_studies/images/arcade-target-dop-100-171017.jpg': '34c99ce5205c99dc88f401e722b94de338cafe715b86e115823c963b9b9658a7',
    'lab/src/deltascope/case_studies/images/arcade-target-dop-100-171018.jpg': '9a6cb75e19800941237e1a2c2c47f77cfdde464f4b9b7894b28e55658b3756cd',
    'lab/src/deltascope/case_studies/images/arcade-target-dop-125-171017.jpg': '02e9691b10bca84bd5367b986cdd4ae9676eac0233a7c18f7bccdc4858782e37',
    'lab/src/deltascope/case_studies/images/arcade-target-dop-125-171018.jpg': '81528d2adecd5eb0e74b6e8a57c6443f3d4a10fc55dae370b631ab659ba547f1',
    'lab/src/deltascope/case_studies/images/arcade-target-dop-150-171017.jpg': '85f1a7a1942c3a665f0900dcf0e9e97bf304092ab432713802b743208c9b31cb',
    'lab/src/deltascope/case_studies/images/arcade-target-dop-150-171018.jpg': '1a9a8a0f9e989a6dd09a6ecbd88b6597d14979516da13f83ae4d439a23838a26',
    'lab/src/deltascope/case_studies/images/arcade-target-naive-100-171017.jpg': 'b7eb14d74c981268c3abf3047579063a195bb607c4ca64802c4ddc292e9db039',
    'lab/src/deltascope/case_studies/images/arcade-target-naive-100-171018.jpg': '8435e84c1523f43f6db09ebd12cb1b1e2f9425aaa81a32b4d73a0c781c4a7dc3',
    'lab/src/deltascope/case_studies/images/arcade-target-naive-125-171017.jpg': 'aed67de8994fc5dba8b8a4a1e6220594a45a03d9778876e4527997aa1fcd9ea9',
    'lab/src/deltascope/case_studies/images/arcade-target-naive-125-171018.jpg': 'c2ddd2a81cde9a036ee3ccb9d63b2d6554f0a751046371e34d39a81a6c029baa',
    'lab/src/deltascope/case_studies/images/arcade-target-naive-150-171017.jpg': 'e77a3077240b8a0a4ff417efc2fe0a7f6f973ff374f7eaf27800145c3e8f04e1',
    'lab/src/deltascope/case_studies/images/arcade-target-naive-150-171018.jpg': 'f61f40d6cd1ac4940f6dd6371f792f20e1c5948f6b49c8628b279045607fe657',
    'lab/src/deltascope/case_studies/images/arcade-target-surgery-100-171017.jpg': 'd2608349260ed9076ac7811a32031eacc742f18c9c7916d5bc1cb42276f0cb91',
    'lab/src/deltascope/case_studies/images/arcade-target-surgery-100-171018.jpg': '49dd501dea72c4880dcc3d996816d31a2517917c5899bfd79399fd7e6e73adc9',
    'lab/src/deltascope/case_studies/images/arcade-target-surgery-125-171017.jpg': 'b8b1d2c8c8598e558878b0321ba7a79f0b84b789d68b42c0e6309591ca3fdc97',
    'lab/src/deltascope/case_studies/images/arcade-target-surgery-125-171018.jpg': '6585fb2a23fde91b639b143d36a516252846425856dbfe12d7ee7057caba6c3c',
    'lab/src/deltascope/case_studies/images/arcade-target-surgery-150-171017.jpg': '1c4adca689b364dd4909c138416d99664aeddb47e8f7bfaf899d0d87559eb4ae',
    'lab/src/deltascope/case_studies/images/arcade-target-surgery-150-171018.jpg': 'e35bae03ca4b43590171a419006eda3c1a67a26a35cad3378dbf201d378b14a8',
    'lab/src/deltascope/case_studies/images/beach-generic-base-100-171017.jpg': '6a283216d2f8a6ccafd607e16882bc7b3c946692a3248d7fc57c9d3dffe098e8',
    'lab/src/deltascope/case_studies/images/beach-generic-base-100-171018.jpg': 'd395fd32fd3fb14ef0dabb11e8911a057fc83bef24a6d4edea91357bee6fd7ff',
    'lab/src/deltascope/case_studies/images/beach-generic-dop-100-171017.jpg': '182e6ea97ef1ba5a4a86f227f1de3c6d885b99e035575bb02681347cb125dca6',
    'lab/src/deltascope/case_studies/images/beach-generic-dop-100-171018.jpg': 'af57a4d080792435375b1bd8187ad3e82c02df1d8cde1d0fe235ddb68ef1a59a',
    'lab/src/deltascope/case_studies/images/beach-generic-dop-125-171017.jpg': 'c6987acb5b6b9f65c4011badf663ee3abc56a6e2aafb6d53203c4fe2c59d619f',
    'lab/src/deltascope/case_studies/images/beach-generic-dop-125-171018.jpg': 'de4382a737b00315dfa753077fac429180577d0ad2fa8d97e6aa8680b81b4d3a',
    'lab/src/deltascope/case_studies/images/beach-generic-dop-150-171017.jpg': 'fe2b5a6b99870e17da995cb7286cfb573e070543d5efbe9688b3b420f079551b',
    'lab/src/deltascope/case_studies/images/beach-generic-dop-150-171018.jpg': '1659101ea266cd9c1235fec395b868d43c7243d0f72bf4071a2c7f405a797e5f',
    'lab/src/deltascope/case_studies/images/beach-generic-naive-100-171017.jpg': 'e56369ca63f4df2430c174497a71030ab57a70b0323f74c66ca4d03da8fc10cc',
    'lab/src/deltascope/case_studies/images/beach-generic-naive-100-171018.jpg': '8d4bff6f830f91e02364dd8a4ba7609eafef845d3f790da00eb03345c8c509a8',
    'lab/src/deltascope/case_studies/images/beach-generic-naive-125-171017.jpg': '8bb2632ef294c095dcf4038e700a8debfb9039ce8fb863975c405b784f73ae36',
    'lab/src/deltascope/case_studies/images/beach-generic-naive-125-171018.jpg': '255043fb45ed87b213a73b1808be2a4c7be1ac8f0fe36a7226de998ff0b4bfe3',
    'lab/src/deltascope/case_studies/images/beach-generic-naive-150-171017.jpg': '6e13bade6989f6ec2d0b6f1575edd6123238684d713a7c4d6807efbdf885d1cc',
    'lab/src/deltascope/case_studies/images/beach-generic-naive-150-171018.jpg': 'c98b659b7ae402e36d7320c214f799b207f669dd108a4fcff427f06468a20658',
    'lab/src/deltascope/case_studies/images/beach-generic-surgery-100-171017.jpg': '176f1bbbd182aab616bff59042b2f0391973e913e5810a4c3fae3c9ff2d5b2e7',
    'lab/src/deltascope/case_studies/images/beach-generic-surgery-100-171018.jpg': '0cd0240d8e2c7f98df9287b1edef4fcccf3fac35e23f942ac47938fc21f1adc9',
    'lab/src/deltascope/case_studies/images/beach-generic-surgery-125-171017.jpg': 'b886dd106cd67c28cb1f9aec853801f851dc77929b1d20c3a706ebba79ebcc8c',
    'lab/src/deltascope/case_studies/images/beach-generic-surgery-125-171018.jpg': 'd666366b1ea25a278c54d18a20a3a0df690e98e35b77cbd7b2122527bfc444c0',
    'lab/src/deltascope/case_studies/images/beach-generic-surgery-150-171017.jpg': 'fdab23dda36fa8b15d3efbfaa44d93237a73848a9cee8c6109167525374bea39',
    'lab/src/deltascope/case_studies/images/beach-generic-surgery-150-171018.jpg': '38cc780af83b1d67da469658a22185333876fbc025b3fe20a334d8d7f2dfa1c4',
    'lab/src/deltascope/case_studies/images/beach-target-base-100-171017.jpg': 'af9c789808738fac028ce7d6f4c6dba22a73c23bda3ecc3296c7e31235d65d78',
    'lab/src/deltascope/case_studies/images/beach-target-base-100-171018.jpg': '075a830cd17c2c4c886fb06afc5c66c2793c81241bc1091d7c2a61810a3eb980',
    'lab/src/deltascope/case_studies/images/beach-target-dop-100-171017.jpg': '47c41e6d37632776d60c65b470674be8387d8b2ddbc3c36690a4fb5dca2242d1',
    'lab/src/deltascope/case_studies/images/beach-target-dop-100-171018.jpg': '88157346a44e1613e378aff1f40225678e7c76af2a2750cfcc00f1fef1c62de5',
    'lab/src/deltascope/case_studies/images/beach-target-dop-125-171017.jpg': 'd48e9e827606e4ff15e26f97c19f5f6ac54b2c2de148838cb3d2f5c87a6561a7',
    'lab/src/deltascope/case_studies/images/beach-target-dop-125-171018.jpg': '3267952002d6aff164aff9b10780bd00fc1ab61bb4c66a6b1c3fee7106b47088',
    'lab/src/deltascope/case_studies/images/beach-target-dop-150-171017.jpg': 'bf49ae031e61c7e85582f87e447e313ad0db45a2469d25fb59578953d2e87ef5',
    'lab/src/deltascope/case_studies/images/beach-target-dop-150-171018.jpg': 'daf4393bd332c5045e2015239766211006301e4a835e7ee813d2e1c63cfb27ca',
    'lab/src/deltascope/case_studies/images/beach-target-naive-100-171017.jpg': '20b53869d22243e8707ab8d93ab6b36b1d48c7bf301fc5813e88d3b762feaddc',
    'lab/src/deltascope/case_studies/images/beach-target-naive-100-171018.jpg': '9049e8ae686674736ac2e734349255ea4faea0591caba894d548a874ad67ed7a',
    'lab/src/deltascope/case_studies/images/beach-target-naive-125-171017.jpg': '484af30809e00e4373e8d799e3a1c0513af8c666a583cca08d563963f0bd716e',
    'lab/src/deltascope/case_studies/images/beach-target-naive-125-171018.jpg': 'dc741dd20965880eaec3916e7703d3a0211f10ccd3b21abe8a51350507726134',
    'lab/src/deltascope/case_studies/images/beach-target-naive-150-171017.jpg': '44032b90174fb790a931eab1ef67c3eca4410be98247cf6cc5253ecf128c8632',
    'lab/src/deltascope/case_studies/images/beach-target-naive-150-171018.jpg': '4bf7f8073d159c5abb792519cf7e148155bc6c3451662c239cd35bc2326d6fac',
    'lab/src/deltascope/case_studies/images/beach-target-surgery-100-171017.jpg': '722d915b5e95e165e37b83c0611cc5acf0884487ae186c76652cafe2bf689f8d',
    'lab/src/deltascope/case_studies/images/beach-target-surgery-100-171018.jpg': 'cff416c836d4b7a0bc3e32ac7725d4588e55c5914ed2f18496696e45d4a42a6b',
    'lab/src/deltascope/case_studies/images/beach-target-surgery-125-171017.jpg': '2aec6811f53c6371dc84f0a6453aa2250e13e8d8639f140973848e1a312b1992',
    'lab/src/deltascope/case_studies/images/beach-target-surgery-125-171018.jpg': '84d5fab1a3ff62c5db3cc8b04c78a73859fcb3eb763ebf7627a3813bb0fbc686',
    'lab/src/deltascope/case_studies/images/beach-target-surgery-150-171017.jpg': '27e922e23d6275cd08877afccee2a92789a0724d571802797f66cefc99732cfb',
    'lab/src/deltascope/case_studies/images/beach-target-surgery-150-171018.jpg': 'e267e7c66ed30f0d227f90e51d2a0ff6f1526b6acc4018a0bf1d826bba345ad3',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-base-100-171017.jpg': '9b8f357fcc0e8f65a01efd20e139556497ce8c81f0c55ab778b70f69c649f691',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-base-100-171018.jpg': '76c3ec8854b8312395d62323fec59c3f6440d833ba8cdfbb34cc3f70e9c9a041',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-dop-100-171017.jpg': 'f24b68d4b3483ba31745f6382be18cc7eb31f5bc796ac338b561206075f1fcfb',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-dop-100-171018.jpg': 'b42cc4e789fe6e9a440323f81ae14e76ea870e8a49b05126584c5b0882d245a3',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-dop-125-171017.jpg': 'e250a5fcec61a90fffc90b690a4d869e2c7254cb0fda5a2bd4ea41a5a86f6547',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-dop-125-171018.jpg': 'c6649edb39cc450e09c263a85ecc7b9f41b6b843567a04bdd7fd9380fae83722',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-dop-150-171017.jpg': '9964a2876a2ea46fc9d28440f770f7b5b5a33bfeeb39548a374cfe6710a312ae',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-dop-150-171018.jpg': '17c639e0dd30554b214c1f0c387f2572957f45e35080ce5036fa224c9ad27837',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-naive-100-171017.jpg': 'a1faf52763b1cd627654e0e5735e38c9fa14e92773d4d7e326b85907e90e0115',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-naive-100-171018.jpg': 'fd94c89dc3e728e3a73ffb6f2856efa21ee91e95be8225b49014f054acce437f',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-naive-125-171017.jpg': '692c93e6b8698716b73d2b19daa41b8c04b18adb1ec59041862299fc80234422',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-naive-125-171018.jpg': 'b73f3d3a40a0f86b11c2bca48fb332ed4a21e6240fc28b9bab6e59d1eaa89dcd',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-naive-150-171017.jpg': 'c6039f916ae3c21a5c2746dc54710a965858900b2b18d4409e4e61c155fc9508',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-naive-150-171018.jpg': 'cb20ae6dda3f13bea79defc956c007d26684c428367a8165e8f0ba5b4a581334',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-surgery-100-171017.jpg': '802c1213270b32efbc7911ccadd40b6f185b7b525db89983616dcb9210cce9c4',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-surgery-100-171018.jpg': '8dc9397c1001511b86ceb680be022ea7eac23f50ed54233336beb814af624624',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-surgery-125-171017.jpg': 'b77b86a4d9fe8a36091119aadc8831c824eec65fb78d58d1767a1a9d6e9cebcf',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-surgery-125-171018.jpg': '2a5507f63049d4d1904bbed891353b1acbe7d6bc8e1fa0340b5a1114678b8c09',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-surgery-150-171017.jpg': '30f267ce1387269d80b8ca843cc31ba36452f25f44402ad86c85cf499b660aa3',
    'lab/src/deltascope/case_studies/images/lighthouse-generic-surgery-150-171018.jpg': '19912d93b94534eb24661dabb00ef22e2b27e13291883cb8473774ebc5d336c1',
    'lab/src/deltascope/case_studies/images/lighthouse-target-base-100-171017.jpg': 'e8182408f035ef7585471cba1cd9cc1839cd40483b43dd128f9286feb57d7d32',
    'lab/src/deltascope/case_studies/images/lighthouse-target-base-100-171018.jpg': '5469edf20976fc3837509c9ae6054a48c27e2798ef10453328f5ad3f0718d729',
    'lab/src/deltascope/case_studies/images/lighthouse-target-dop-100-171017.jpg': '3700d9bba23e250b49c8e6a22b8de2f2b796da18972ccc1bc00c5356b800d569',
    'lab/src/deltascope/case_studies/images/lighthouse-target-dop-100-171018.jpg': '0a72517ce3e3eb4a7f49302c7361bf177f050bdc6b97a5d2d838655f33037352',
    'lab/src/deltascope/case_studies/images/lighthouse-target-dop-125-171017.jpg': '77ac658cacaa3da93390be01ce45a6c5edb702ecae44f75a9f9b8a6889071bf6',
    'lab/src/deltascope/case_studies/images/lighthouse-target-dop-125-171018.jpg': '441804e59213f43c20e208cf595ef3bf340accecbeddfebe3ef5fbdedaf1eec4',
    'lab/src/deltascope/case_studies/images/lighthouse-target-dop-150-171017.jpg': '7788a45f124ff645839e7f86da7d37c3ddf852047f72d9811144f934e7ff090c',
    'lab/src/deltascope/case_studies/images/lighthouse-target-dop-150-171018.jpg': '70cb368357a9c7e4a43c395db2d9f4cdfb19032d31ceb16487eee8e421be056a',
    'lab/src/deltascope/case_studies/images/lighthouse-target-naive-100-171017.jpg': '951b60351005db187d2ed014177e371a9140d79d9baa763554faa4662fee8228',
    'lab/src/deltascope/case_studies/images/lighthouse-target-naive-100-171018.jpg': '101e46b347743fa06ddeca1648f55631701c3007ee0dad77c3c56def526c9cc8',
    'lab/src/deltascope/case_studies/images/lighthouse-target-naive-125-171017.jpg': '9bfcea0ea13cc1b796ee2e0db29d5ac02e5bfcd7b53dc781685752a2d97b4cf2',
    'lab/src/deltascope/case_studies/images/lighthouse-target-naive-125-171018.jpg': '1784d15b01a9986ec67fc3fc7d2d8d542bfcc09c467fba3f3b5a3d5bdc7edf70',
    'lab/src/deltascope/case_studies/images/lighthouse-target-naive-150-171017.jpg': '1bf0b5befecab34f48f08c638e355a34f89a95236779ea124eb6d1668d5555ac',
    'lab/src/deltascope/case_studies/images/lighthouse-target-naive-150-171018.jpg': 'f29954c274bc33b38fdc8db9beecda5d0578e5d33e114dbef4c7e98000001633',
    'lab/src/deltascope/case_studies/images/lighthouse-target-surgery-100-171017.jpg': 'ec861eff076cacf5831501844db8b2f21574646b8fc9e6a43af7cc61a984c74a',
    'lab/src/deltascope/case_studies/images/lighthouse-target-surgery-100-171018.jpg': '2ddbb5aa810c394bda54b486f395ef341dd4dc1cd936caa9441f3c264910b63a',
    'lab/src/deltascope/case_studies/images/lighthouse-target-surgery-125-171017.jpg': '238381868ae5d8ada08b13802e485583889348eaf145efd249a366bac4fcf5db',
    'lab/src/deltascope/case_studies/images/lighthouse-target-surgery-125-171018.jpg': '767c2c0cb9f3510bbc8637bfcfd3f1703dd1683bd6b29860d1f51b817b9187e1',
    'lab/src/deltascope/case_studies/images/lighthouse-target-surgery-150-171017.jpg': 'c5b2d0633ddac7b8d073e7664477b7a5e1107d886d5b321d18e3fc3a432990aa',
    'lab/src/deltascope/case_studies/images/lighthouse-target-surgery-150-171018.jpg': 'cf64ce1d8bf31301387ace6c755320c1acc813dd45f6ffd93d0fe2de3ef813a8',
    'lab/src/deltascope/case_studies/images/space_station-generic-base-100-171017.jpg': 'b68f0485a640faa562415fdf708ace10f43ebfd6c8559b64f011d0205b1ee218',
    'lab/src/deltascope/case_studies/images/space_station-generic-base-100-171018.jpg': 'd33ee8c7802cd8108b35679324066981f4e369153e582f50357e0a3f3f2c142c',
    'lab/src/deltascope/case_studies/images/space_station-generic-dop-100-171017.jpg': '5c593c9af02bebb14a989d4428a8e8817348595f7349cad2df5fe21f0861f588',
    'lab/src/deltascope/case_studies/images/space_station-generic-dop-100-171018.jpg': '529c9ac1bea337b99738c85f2f0119a364381462f361a90fae7c081054ce47af',
    'lab/src/deltascope/case_studies/images/space_station-generic-dop-125-171017.jpg': 'c0112bb7e5dd724c6272997f72278911a25a2de5f8ff997eb3529705d8bfa14d',
    'lab/src/deltascope/case_studies/images/space_station-generic-dop-125-171018.jpg': 'a56ec4730c5e7efbae5a3f322d71aae390b9c2dbf9e1f00a4b969f6c091bfd60',
    'lab/src/deltascope/case_studies/images/space_station-generic-dop-150-171017.jpg': '9eab13fad75fb40980096ecc9a5c34372a43f5bacdac4fd2e393ab2995e9f618',
    'lab/src/deltascope/case_studies/images/space_station-generic-dop-150-171018.jpg': '4fcc656600470eda9a88d208d67fdb349c0c45ab5dbe71654583c52201688b6b',
    'lab/src/deltascope/case_studies/images/space_station-generic-naive-100-171017.jpg': '1286ffc0b8fbfc3e8b6e83e25aafd44205611dc7bae4ff3e959b24b8e6e64208',
    'lab/src/deltascope/case_studies/images/space_station-generic-naive-100-171018.jpg': 'a849f11521dc763426c094b0404b21c1b15e2aa68f60012627fa421337b6ed2d',
    'lab/src/deltascope/case_studies/images/space_station-generic-naive-125-171017.jpg': '0671f7fe3fb25f84bb5c4d944d0bc73f6111aa0134d314686d41125ce4d77ae0',
    'lab/src/deltascope/case_studies/images/space_station-generic-naive-125-171018.jpg': '8fd093e321d019b15cbd6e6db21e382a727522b555ed72dddb91ec0724988a9d',
    'lab/src/deltascope/case_studies/images/space_station-generic-naive-150-171017.jpg': 'ba85f22d87c5a7668e8192a1a6c5c20b6af23d00e9ec838a5bdf7afe1bfcd411',
    'lab/src/deltascope/case_studies/images/space_station-generic-naive-150-171018.jpg': '257296abf49b9de57943889bb35c250741edd1b16bb0693f378c577c12936aed',
    'lab/src/deltascope/case_studies/images/space_station-generic-surgery-100-171017.jpg': '4a2c811c4fe9ee6f7c70f414cd49894d1a456ea2a232689819398d1929e59ec0',
    'lab/src/deltascope/case_studies/images/space_station-generic-surgery-100-171018.jpg': 'bb7d60ef7dcd548f0b350fcc86210b6b7682b368926986535d41d4ea73e9c1d3',
    'lab/src/deltascope/case_studies/images/space_station-generic-surgery-125-171017.jpg': 'b25303620f2b33a5285b507c502dd69af8c5fd61296fccc89855c12a104f0412',
    'lab/src/deltascope/case_studies/images/space_station-generic-surgery-125-171018.jpg': 'd8821e062c2810f833ac1986abf9f31914591971ca503c523cd27315e54719e3',
    'lab/src/deltascope/case_studies/images/space_station-generic-surgery-150-171017.jpg': 'bf96ea772e82d40ff911b1bf5166ca9add886b2b146bc0fd05a85c6b30920d6c',
    'lab/src/deltascope/case_studies/images/space_station-generic-surgery-150-171018.jpg': '9768c0acf9c8b60e31d5b838b210e9a4c368daf917020acbac0683657f51ae64',
    'lab/src/deltascope/case_studies/images/space_station-target-base-100-171017.jpg': '0c4d9d1c48cc08448f20a135130624b5d536242914f81bfe98e5b38e8b52d00d',
    'lab/src/deltascope/case_studies/images/space_station-target-base-100-171018.jpg': '8e6d693f493281a4e5b49ab6f50f899f8ffcf09a2debb7bbe682974bd6adf3d8',
    'lab/src/deltascope/case_studies/images/space_station-target-dop-100-171017.jpg': 'eabc419fbd248bcc41359a35017b9d3bfe3f85e21e6a742044d761e73437a988',
    'lab/src/deltascope/case_studies/images/space_station-target-dop-100-171018.jpg': 'e79020120d5b50c11be8afd7dfe89dc102f9945d109b52f057d981fbe82da3e8',
    'lab/src/deltascope/case_studies/images/space_station-target-dop-125-171017.jpg': '0c4191550b7c966b12aa73d47bdd415a6daf5240c5cc5098026c6a6dedd44dda',
    'lab/src/deltascope/case_studies/images/space_station-target-dop-125-171018.jpg': '96a41f706008522d94c27bdb6190924c8c9b7bf74a631fdc608531cd3fb6eeda',
    'lab/src/deltascope/case_studies/images/space_station-target-dop-150-171017.jpg': '6516471db9719076af3e2ad53dfe18ac4a3945b3c08f0ec16dae6e15a0c94b92',
    'lab/src/deltascope/case_studies/images/space_station-target-dop-150-171018.jpg': 'c278752296da94630a404b395e98e04a127080689b300d162a3020704369e9c3',
    'lab/src/deltascope/case_studies/images/space_station-target-naive-100-171017.jpg': '4122450913948cf313e5c3192a1ea6f948e205759d668ca4b60ccd8457232141',
    'lab/src/deltascope/case_studies/images/space_station-target-naive-100-171018.jpg': '78f3113ba7e7fabf94bf3c86766e774f4dc3e20f45299140c4082713f4734cbd',
    'lab/src/deltascope/case_studies/images/space_station-target-naive-125-171017.jpg': 'd6426c2c394208427e9368e0e60007e0428ee0a30527b32cdfc414cc4b520ed3',
    'lab/src/deltascope/case_studies/images/space_station-target-naive-125-171018.jpg': '42881064bf597b166f1b81e1cc1781e66d3ade6ef769065685c4e177c48c4970',
    'lab/src/deltascope/case_studies/images/space_station-target-naive-150-171017.jpg': 'ce0bfce533149077d772149c9d21723eed910d015edcb4084c7cf92858f631d2',
    'lab/src/deltascope/case_studies/images/space_station-target-naive-150-171018.jpg': '1ff90a69aeb5a23dcd05e7a8c4091794fd129f4f4e83170471fc79cdeaa07ff9',
    'lab/src/deltascope/case_studies/images/space_station-target-surgery-100-171017.jpg': '8cc7827e4a5d4b5cbb959d3513b71fa82f544054e67b866fd609dc7b700bc39a',
    'lab/src/deltascope/case_studies/images/space_station-target-surgery-100-171018.jpg': 'e486f21792311c9136a7c008c6329f9ddade4c6edf0fb4cf49181cbc712678c5',
    'lab/src/deltascope/case_studies/images/space_station-target-surgery-125-171017.jpg': '8164e94488c965c0869f7e49f785a9c679dad4272f4cdc45ec1f7e9cc96b51b1',
    'lab/src/deltascope/case_studies/images/space_station-target-surgery-125-171018.jpg': '8a8536ac3a9a06e125e7ab23dcdb64e2bba9ef2b09e48c94ae5ffb31ec764961',
    'lab/src/deltascope/case_studies/images/space_station-target-surgery-150-171017.jpg': 'fc976db64336c5a9ad51c05d47ab9bff78fa8b1128c7982104c97e7af4d73745',
    'lab/src/deltascope/case_studies/images/space_station-target-surgery-150-171018.jpg': '9f307f57c19940750279d5084cf815f6b82a06ad0c2f26accbdc8130e773c85b',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-base-100-171017.jpg': '1f5f716f12df21e3e22a7af0c2640759d0c47c0dc8a998d181d9edad87820d87',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-base-100-171018.jpg': 'd15c27ecb0bf2f1a7a70955cfc741db22eb7cef756bbbbf287931ba38fc2f168',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-dop-100-171017.jpg': '6a2e81b542e733ef014524d3e78e7b8cc9934c3f7bb865449a6ec8979478e48a',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-dop-100-171018.jpg': '96ea723a3e40498e81ee457f5306a4e90ece0d6b7567b80417ec3a1fa4901c25',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-dop-125-171017.jpg': '7247c8c88378f72211be5999b29b5af661272160352699d651118036619086b0',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-dop-125-171018.jpg': 'af506c3f89bb5e681e8a1f44245893a64f9550c28eed5ed011394bb76cfe04f1',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-dop-150-171017.jpg': 'de2f63a2dd7421fcaa2f4e6adfecd570894ba5fcca596c8eba9e02d4200e1bd1',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-dop-150-171018.jpg': '4689a6309590d4eef06c14cebda1fd85a7cddd96921485f617f7e509b9e527ec',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-naive-100-171017.jpg': '46bfeb6ac6c76fed3883d3b3a808a7c54bbf061519985ed0e006bd274351d6df',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-naive-100-171018.jpg': 'a4cc1d342836ed1003cb1c9a9fb869b1e56934d39e2073aaeb5605a954445fe9',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-naive-125-171017.jpg': '7acfb61ff303bb3b96e36fecb258528016196883a0ce3d00127dda7fef7d5221',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-naive-125-171018.jpg': '40b99f78e11011e19bafdee30dcf13d7e8980728e61a53f7844449f9e77bd2b9',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-naive-150-171017.jpg': '251c4d9d2c501583c9208cd7096cfe216479d5860d316817f66daaea4df10a2c',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-naive-150-171018.jpg': 'b4e99440a1f08203821447c383022ae666dd607fcf083cd47ad34d4809fbc22d',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-surgery-100-171017.jpg': 'aa6f3098c188749a55334bd54b50dfad9b23236223ffa4baa9e4554b5ded85ed',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-surgery-100-171018.jpg': 'def2a03e454d0f907fcf42b43d94bef4d4c85b9957fd071a461e395c49833b7d',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-surgery-125-171017.jpg': '01abf69fdd62a91b73efbf74052442acd17e08838e211b77598c241f1fbb3f39',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-surgery-125-171018.jpg': '4acfb7e8fdc633efd43b6588f93e91fa0a61989673385955c505492cef0f24cf',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-surgery-150-171017.jpg': '21f0b56a62593e212e07bfd14fde06af1ed1d48f79315c5da69348f5b36b379d',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-surgery-150-171018.jpg': 'bda58abbc69481a7cadeac994a2fc9a7ba0dc2e535b01325405dafa66e7f9c9a',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-base-100-171017.jpg': 'd76a36c2ca7feac25042ffc5f14b6e87437cfc3871529951ed51adb2593c491d',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-base-100-171018.jpg': '8aebbe11d4f5679f9ccb373f82ae4cc1df23143c282670257471bf4c5c64fee3',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-dop-100-171017.jpg': 'd0969ed2bc431d5f93a946895adebb6719174127ba964d3e0e9aeedbaa521750',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-dop-100-171018.jpg': '9fcb5823ab6d73e5097057947740f21e6f7553a7cc42c3c0a4ac8e163f6a3e26',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-dop-125-171017.jpg': '8b8c3ca25a080948766a85c63733e3b87845d67a33d18d52b6f7c857135f11b3',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-dop-125-171018.jpg': '4c26f28e7d88e6e797bd192dc589f46c27b79edc940f6f5e8eb2b91aad06c047',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-dop-150-171017.jpg': '95be544aa0ac191c57f3ac280400cab9fe1bd92aee35c10c5617bb71ec4d45eb',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-dop-150-171018.jpg': '90240efd0ec514b04d3ba7492e49c50861be96c45a8f00970c7cb29d9b91018c',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-naive-100-171017.jpg': 'e7f4675087638a7b2bbf7f63f3996be0da80f9b9ed20d5692a3a3f54097c57fc',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-naive-100-171018.jpg': '7bd36a088f4dc0b2888c32b5d0195bc3670aa67267ce9ef92ebfce89d4ee2e7b',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-naive-125-171017.jpg': 'ac1d84e4512f4567ac368879a72ec0048125dc6bb0381de0247bd35b64f69ed7',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-naive-125-171018.jpg': '76245e3c433cb57dff850f3db3e1449182c1bea8fc16878b16bb582cd50cdf50',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-naive-150-171017.jpg': '8d5c7031fb68ca5af25bac8672fb7540f13a14b01aba20f9f9bcb6c5c0a95b36',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-naive-150-171018.jpg': 'f88c1f2b572a107cd7cb74ba4d3df3703063a5d8b85d738b99148a2f15f1d24b',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-surgery-100-171017.jpg': 'd845bdedc0dd9a178a9f0fc46007ffeeff0bd654dde12224be5f15f03751cc8b',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-surgery-100-171018.jpg': '68f3ad8f9dcb8b6c0e7aad254ccc27d2f3b6f5bd93056fadba1b0d14794d6a9d',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-surgery-125-171017.jpg': '3f49ec3fc6c8c7bb892ce18f520f760baf8f7bbd057589c9cce1fc4aae02d02b',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-surgery-125-171018.jpg': 'e4eb3bdb16b6e4327e5b96553a70c9998a3c2d0880f02e235f05a7a655edee70',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-surgery-150-171017.jpg': 'c53f8f8bb0ea1f49c2f656c98a608680fe3ab50e0635126abd25d9cdd5ea6565',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-surgery-150-171018.jpg': '9fdfb188b90024cb6bdf218b3d8108088b6bffba6647c61a2272492ed88799d9',
    'lab/src/deltascope/case_studies/images/beach-generic-a_side-100-181901.jpg': '2d348c9e409ad57c402612febb7de8d49fdd1e7fc7ff7fb21e6f29669a95a4e3',
    'lab/src/deltascope/case_studies/images/beach-generic-a_side-100-181902.jpg': 'ea9d5c0ead9bab05831924a628cd5e55197919e716c1d58db4b49c532a2c592a',
    'lab/src/deltascope/case_studies/images/beach-generic-a_side-100-181903.jpg': '2c09f1be75a7203240b1b847f459a92540779a90bd406d0a9e72508bc1295ee0',
    'lab/src/deltascope/case_studies/images/beach-generic-a_side-100-181904.jpg': '9ee229eeb8041f4249de1dc160f7b99aea627eada82ec753d97d8c7323c65010',
    'lab/src/deltascope/case_studies/images/beach-generic-base-100-181901.jpg': '7e9007dc3563d96aa1aed52b7ecc28ac5188208c0d91edb3ab242c59220fde61',
    'lab/src/deltascope/case_studies/images/beach-generic-base-100-181902.jpg': 'dba3042edd86c171b8b74fc159431564c2c0a738de4e158dbfdab42a81b2ded3',
    'lab/src/deltascope/case_studies/images/beach-generic-base-100-181903.jpg': 'a6c85d5554d9413d86774d9d1b7f682d7f1df425c4c978717d6a9dcb3bd9ab48',
    'lab/src/deltascope/case_studies/images/beach-generic-base-100-181904.jpg': '0e9e907873d6e475d315656a486999f8de34c228e13e576d01e2a3665a698e7e',
    'lab/src/deltascope/case_studies/images/beach-generic-naive-100-181901.jpg': 'a0f6e8c424481b22534acad04623a0bda66f17d1063400da012fa9f18633e25c',
    'lab/src/deltascope/case_studies/images/beach-generic-naive-100-181902.jpg': '2d1c67247f909630be4e367e93cbbd6d6b0f02f7c811496e3ac591e79d80ff7c',
    'lab/src/deltascope/case_studies/images/beach-generic-naive-100-181903.jpg': 'e2e7d9697cffbc4ea898ddf35f60125909472331580bd7448b20f6daa1f046b9',
    'lab/src/deltascope/case_studies/images/beach-generic-naive-100-181904.jpg': '2c7933bc5f4c10113a91138be26034c22702f37d34af733342500280c676121d',
    'lab/src/deltascope/case_studies/images/beach-generic-trigger_amp-200-181901.jpg': '7a8d544a795d334f1a3542fd8e9f2120dab27630d2d0a93e1cab6a233a98ae86',
    'lab/src/deltascope/case_studies/images/beach-generic-trigger_amp-200-181902.jpg': '756c608f3179ddaab566eb4d4bbe23e87e436bda8a7bba2576e46216526fa573',
    'lab/src/deltascope/case_studies/images/beach-generic-trigger_amp-200-181903.jpg': 'd44d2fbf9d53513a2e0bedccb4f120f93ff09a4da9542d78069cbe6c75eefea1',
    'lab/src/deltascope/case_studies/images/beach-generic-trigger_amp-200-181904.jpg': 'e84b5ecd171945698e10514d6520509457786afe455db1b071b002033cd37eed',
    'lab/src/deltascope/case_studies/images/beach-generic-trigger_amp-400-181901.jpg': 'ce8ac061f70ea1b3b9b54bc257e90946703958db13bc2579101711c39f2b34bd',
    'lab/src/deltascope/case_studies/images/beach-generic-trigger_amp-400-181902.jpg': 'd0ca69777fe96b233e524d17568aa5d81518a163433f227e93d640ac60a6011c',
    'lab/src/deltascope/case_studies/images/beach-generic-trigger_amp-400-181903.jpg': '8310b56d4e2b852313107131af9aa936be80baf39f047c2f336a7ba364cf0e33',
    'lab/src/deltascope/case_studies/images/beach-generic-trigger_amp-400-181904.jpg': 'b8b759733f0bc342a8a6aea439f0e15984e9f671a070389e0ca4dc6cf8e54521',
    'lab/src/deltascope/case_studies/images/beach-generic-trigger_amp_peel-100-181901.jpg': 'ac2747368a0a148cf07fd7e46a6b2cfa56e9ab28af8d15384611ef05b73a4065',
    'lab/src/deltascope/case_studies/images/beach-generic-trigger_amp_peel-100-181902.jpg': 'ea2115695eaef5e1ef2b726317dea068236797598c365eb5dc50b88884ce848a',
    'lab/src/deltascope/case_studies/images/beach-generic-trigger_amp_peel-100-181903.jpg': '08c68c2813a30b283c01a9fac1948dc0e43fcfcbffcf89ed54013a21bbc34218',
    'lab/src/deltascope/case_studies/images/beach-generic-trigger_amp_peel-100-181904.jpg': '1969e3ae5f3379ce6ae79a486017b998c5da515f6a63ca97ff85cba6a540f0f2',
    'lab/src/deltascope/case_studies/images/beach-target-a_side-100-181901.jpg': '49567a1f6c904ecc2b1f7275e9481cd82ad4d5002c3156dab92dce38d75ca1d1',
    'lab/src/deltascope/case_studies/images/beach-target-a_side-100-181902.jpg': 'beb334fb5ea4f3f0be32838a2e80edd0575104a9ed66b8b7ddc3de4071a55be3',
    'lab/src/deltascope/case_studies/images/beach-target-a_side-100-181903.jpg': 'ab2f3d787b9f8df820a343ca8e6fd4bf2d68dcd7028f7568c7b463502a7d4834',
    'lab/src/deltascope/case_studies/images/beach-target-a_side-100-181904.jpg': '281c55394155ebe0c11eedef28ff8624ba1bfd167a9eb4b364e453d116e6c1cc',
    'lab/src/deltascope/case_studies/images/beach-target-base-100-181901.jpg': '7b444aa794749c912f6e40822df3e6e6dcc5b674712acf9e26a269788f9d5c22',
    'lab/src/deltascope/case_studies/images/beach-target-base-100-181902.jpg': '6361ea64e3cf6231c6f2096453e33cd340adec4fe2f71d742b3f65185b9455a6',
    'lab/src/deltascope/case_studies/images/beach-target-base-100-181903.jpg': '6bd29e43a6304bfc14391d6659725ebd878d56819b0d7b4badc1b810da807e3b',
    'lab/src/deltascope/case_studies/images/beach-target-base-100-181904.jpg': '02df18fa2c4bed04495751a9b0f381cc2f393fcf5385bccc1ed1c10b8f61c38e',
    'lab/src/deltascope/case_studies/images/beach-target-naive-100-181901.jpg': '029d3102480b5ca30c642b63a4c6199dc9f53fe66b51373d5d4207b41ec7d94d',
    'lab/src/deltascope/case_studies/images/beach-target-naive-100-181902.jpg': 'b993cd2238035a8bcd6ffd0e890473587c04c9b45c1905e175fd8e95d5a80e50',
    'lab/src/deltascope/case_studies/images/beach-target-naive-100-181903.jpg': '7d59fffd129136628f4d8549e4b3e502c55762aea99c22e0495efcf641c55ece',
    'lab/src/deltascope/case_studies/images/beach-target-naive-100-181904.jpg': 'd28a4c90a41833c10adb37c5bb96648c82695eb273711d6db578d90f310359b1',
    'lab/src/deltascope/case_studies/images/beach-target-trigger_amp-200-181901.jpg': 'd05f37b73440a23959fb4e45172b811749b51ddae75a79c0a34b377fe5f781e6',
    'lab/src/deltascope/case_studies/images/beach-target-trigger_amp-200-181902.jpg': '08e1fc0f77fde4ce9e28d26bb7f76f4d04fa92792f6040167b4241f53069772f',
    'lab/src/deltascope/case_studies/images/beach-target-trigger_amp-200-181903.jpg': '9243f177982093e3a44b14c1fe5d363ab33062134509272cc77250a3c19d1bd4',
    'lab/src/deltascope/case_studies/images/beach-target-trigger_amp-200-181904.jpg': '66d97d22da37c59fad0adac9251efe1950f67865e348f0f5d5b450f635f4fc98',
    'lab/src/deltascope/case_studies/images/beach-target-trigger_amp-400-181901.jpg': 'f77bfe5cd80cef6eab84ee2dde4bfd6071a8cb2b49c85ed8163fc2c7fc53d419',
    'lab/src/deltascope/case_studies/images/beach-target-trigger_amp-400-181902.jpg': 'd0172107a7d4254fff1ad5b11041aaab1a0fb511f98fc90837169bbb42c31c1d',
    'lab/src/deltascope/case_studies/images/beach-target-trigger_amp-400-181903.jpg': '9bab19568ca3ef61c65ffa4e182a74ed877dbc48569717e194b0bf03775ee10f',
    'lab/src/deltascope/case_studies/images/beach-target-trigger_amp-400-181904.jpg': '5350493cf7e1696767b2f827674dd824beb75ec9342fb3e904a368220b71abf9',
    'lab/src/deltascope/case_studies/images/beach-target-trigger_amp_peel-100-181901.jpg': 'e3ab67fee2573ed5650cd39f2be12c5a999acfa4fdc2265cfe1d7f9dab34811d',
    'lab/src/deltascope/case_studies/images/beach-target-trigger_amp_peel-100-181902.jpg': '3fa9d8a8694d6a2185a9ca83a264197b4217694fef3e84bda0d73c2a9304cbce',
    'lab/src/deltascope/case_studies/images/beach-target-trigger_amp_peel-100-181903.jpg': '507bfb1845b9faed7dcabb3b3445573416dc059ff3a1ca9550c4732bb45690ae',
    'lab/src/deltascope/case_studies/images/beach-target-trigger_amp_peel-100-181904.jpg': 'ff24450fcec8f3b6ae8b9a7ea0f521da68f00b3b8649aed0dd770e9dc05429dd',
    'lab/src/deltascope/case_studies/images/space_station-generic-a_side-100-181901.jpg': 'fc44e7934f3e60925fe1d1a2923ffaa0fdec76ee294ec9cc2f0e6a10536d5f39',
    'lab/src/deltascope/case_studies/images/space_station-generic-a_side-100-181902.jpg': '7f1f731509771b51eff87661d262dc2b322a3b06f007884453d7fbaf81af71c0',
    'lab/src/deltascope/case_studies/images/space_station-generic-a_side-100-181903.jpg': '14227ce29bc226f53aeba218b1cff0e9fb54254900ccccd85d8f8649f20ba8cb',
    'lab/src/deltascope/case_studies/images/space_station-generic-a_side-100-181904.jpg': '9091b6c5c9659aa815ca8b33e2ee6a64a4125c2f76ae2566792ecf781f6722dc',
    'lab/src/deltascope/case_studies/images/space_station-generic-base-100-181901.jpg': 'e8998a30445f59eebc6a311cd5d2703ace702b1928e6da97efc45be29e124e05',
    'lab/src/deltascope/case_studies/images/space_station-generic-base-100-181902.jpg': '9492fdbb3052a5491f9c6b138c06eeb630e8328a8e59091cd52d76df6c512986',
    'lab/src/deltascope/case_studies/images/space_station-generic-base-100-181903.jpg': 'd630ec495b03b3068735439fc087c095e57561cbd5693209e1b4e941417fe298',
    'lab/src/deltascope/case_studies/images/space_station-generic-base-100-181904.jpg': 'd45979fcceb76d500cec8ac4f79f1fe69f7e4d549ad9c6fb260137debbbcff8a',
    'lab/src/deltascope/case_studies/images/space_station-generic-naive-100-181901.jpg': '5d587a6048b9e0cc358c52f13080bb5310de949ecd47f6141318529e2c59355f',
    'lab/src/deltascope/case_studies/images/space_station-generic-naive-100-181902.jpg': 'e44b22ec0f3163fb6161e6cb8d65b2b11a4784560c25043601e242cba72a3e59',
    'lab/src/deltascope/case_studies/images/space_station-generic-naive-100-181903.jpg': 'd461f48fc9377ab28189c1459130bcb82217cff627555cc7eb909fa66e5ad64a',
    'lab/src/deltascope/case_studies/images/space_station-generic-naive-100-181904.jpg': 'be0409e751bf4cefcd94ba173c30b5ac8557e1240f2892696886b3b7e8a66db4',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_raw-200-181901.jpg': '76bd3056d4c25e792a19bcbee0f3901edfeaed9a264a2fd92c97ea8175966b2c',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_raw-200-181902.jpg': 'f564ecbd6fa96701f2c2203d6ca2dbf2c8f9e1c93dcee0fc44402da17149f83d',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_raw-200-181903.jpg': '2bee746b1c0d51a02dbf8fe45f8a28678942d788446a31248d858756926c70d5',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_raw-200-181904.jpg': 'a66230e4c6b62dcc9677abe8b625a30ff5b2e788420f1ceed39f6ff2eb8af88a',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_raw_control-200-181901.jpg': 'aa7adc24d5d2ef3fa572ce2cf38a0af1bb2854a3c50719ba8d8c83e521d57648',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_raw_control-200-181902.jpg': 'e0ecf3f588849482e8368cef3751de9ff49f0e7ce0c24c854d918c55e6be9681',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_raw_control-200-181903.jpg': '5c51dba233450d8c58c59fee889c10bbf961c0ce763cfe2e780b5ee6c38d9290',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_raw_control-200-181904.jpg': 'f5bed4c85d8366fa0fdcb58e909b2df55f1fd068b0ec1e6957839997fddf7b0e',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_specific-200-181901.jpg': '40f40a5583d7637b405f1d8f59a9c0a90269e0a7a28ce65ad32200dbd06fea59',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_specific-200-181902.jpg': '5d7b2ef840192d8fac28579d44abd95bce040c0936d9fa8b94398ca0619c64b4',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_specific-200-181903.jpg': '7af5f0eb3a29c58d3b41e8822150f5c2adee418aa416c078921c7aa85975c480',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_specific-200-181904.jpg': 'b6c9c92e3619eb06c723466f8b438a511ccdf2e902391378187afd0219b3fb02',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_specific-400-181901.jpg': '8593d1e3ef6c5e10a79dcc65d8ae29a5a0ca4ea7e2d2f4bddef77c7decaff8d4',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_specific-400-181902.jpg': 'c1b1502d0269d7c5998a8c3d5923f01054f0e6f8083f6ee7a4caa6e3129ca3cd',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_specific-400-181903.jpg': '6b84dcb2e84ca41d5bd12fd190ce9918e0f0bc871b999365902ecdf21fcd51f9',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_specific-400-181904.jpg': '7e4d5a32545d0cb399e529eef12eb9fa02e1bc04a68ca010dd4e31d6ac869f16',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_specific_control-400-181901.jpg': '566582282fad0d1b0e5865cf4f60967f1c5c2464937dd9cbe0ab2fb16ea144fb',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_specific_control-400-181902.jpg': '371875a42550f79228707fe245a70e3508c65dd2144cecc4f452deaa0804679f',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_specific_control-400-181903.jpg': '2a97b8ac3cb2e37ec20d1003b5ff0a62bc868be5ae86eff30904c3c047a6e18d',
    'lab/src/deltascope/case_studies/images/space_station-generic-peel_specific_control-400-181904.jpg': '54ed90935c524f81317944c13a603c8b51523925b1e46e9ced24d5f2c731afcd',
    'lab/src/deltascope/case_studies/images/space_station-generic-trigger_amp-200-181901.jpg': '207b3f0ed0ad936465fdf5b4d9642f8502f81a6044a4e08e27122044f5cc50b7',
    'lab/src/deltascope/case_studies/images/space_station-generic-trigger_amp-200-181902.jpg': '78e1d3c370c269c4524f089ecd960cfce65d4b2d06ec8d7ee51ce9a60d170bd4',
    'lab/src/deltascope/case_studies/images/space_station-generic-trigger_amp-200-181903.jpg': '455ad5d0601e1f713434ad87516a77f7f723e7a4d13bb1c9b908b33e7951f0b4',
    'lab/src/deltascope/case_studies/images/space_station-generic-trigger_amp-200-181904.jpg': '283dcb43172690202d8256b62e18f69f258dd2a754ea41e1022df94d831d01cd',
    'lab/src/deltascope/case_studies/images/space_station-generic-trigger_amp-400-181901.jpg': '978c5c543be41188d22ca0ffbc01ee6402bdf405d6daa2fcea125675a9fd200c',
    'lab/src/deltascope/case_studies/images/space_station-generic-trigger_amp-400-181902.jpg': '168fc498fa45e6af93e6bad884d63acfc9170646fec531a5f18b1f5b28e9ff3c',
    'lab/src/deltascope/case_studies/images/space_station-generic-trigger_amp-400-181903.jpg': '5db8c40708b0b2016d3fa8ccb5665488fe3a1a03381ee22843d58702f4a17b95',
    'lab/src/deltascope/case_studies/images/space_station-generic-trigger_amp-400-181904.jpg': '649d9e48d1fd034abf5e724c2402ea0b5c796cddd65c74134a4f8d80b5f1315c',
    'lab/src/deltascope/case_studies/images/space_station-generic-trigger_amp_peel-100-181901.jpg': 'ec6bf29a695a91609697657b26a2fb2cf4daf4dc3e40826effd520fcf45ff132',
    'lab/src/deltascope/case_studies/images/space_station-generic-trigger_amp_peel-100-181902.jpg': '2932e0edd2abe33ee9553e8e8c0d603678e2c08f5faf1ae4b7df68de165958ec',
    'lab/src/deltascope/case_studies/images/space_station-generic-trigger_amp_peel-100-181903.jpg': '61ea95da0a6b72d5b2742d16394ba9ce3237118ed60699454fccb68758aa058c',
    'lab/src/deltascope/case_studies/images/space_station-generic-trigger_amp_peel-100-181904.jpg': 'd0b13441fc082b5a933c86aa6d0d93124650ba244500c21a04c5c4ced8754eae',
    'lab/src/deltascope/case_studies/images/space_station-target-a_side-100-181901.jpg': 'b8dc3f0064e94910916bf8c471ba61894ba6c0de44213c7a440b87a2bfd9d398',
    'lab/src/deltascope/case_studies/images/space_station-target-a_side-100-181902.jpg': 'bf95b055ffbb9c7c85768f52c78818d0ed4ead1963f835aea52e0f95920b3076',
    'lab/src/deltascope/case_studies/images/space_station-target-a_side-100-181903.jpg': '7790ed0792a159b462dfac72ad03abe146c1e8c31c40ea2b231dbc9cd7164d3c',
    'lab/src/deltascope/case_studies/images/space_station-target-a_side-100-181904.jpg': '1d48b96d2df45cc799d670aa0b9476c457087c56df9cb4b2ab8d8656159d4269',
    'lab/src/deltascope/case_studies/images/space_station-target-base-100-181901.jpg': '9821881588faffd2411e93d3220cac4009d6808e705c7148bafc277ef53f9e9a',
    'lab/src/deltascope/case_studies/images/space_station-target-base-100-181902.jpg': 'c9d7abca74d866dbe316dbfb3317168b663da200bc59a323e6c76ede999743bd',
    'lab/src/deltascope/case_studies/images/space_station-target-base-100-181903.jpg': 'ee9df85c9ce14a24877f91c46e89b9aca496715adaee8fef423544267ce0092c',
    'lab/src/deltascope/case_studies/images/space_station-target-base-100-181904.jpg': '61122a194961011559c0220821b88ddc7170f548a3255cb607c4057fc50ea57d',
    'lab/src/deltascope/case_studies/images/space_station-target-naive-100-181901.jpg': '20d8371bea608c0542815c72cc9b20f23da7832bda2784378d08e2941ab4a6d9',
    'lab/src/deltascope/case_studies/images/space_station-target-naive-100-181902.jpg': '60a83b5b8afb871d69a9c05021c417096d95e45342b83ef6b9ecdb591d9dd9e8',
    'lab/src/deltascope/case_studies/images/space_station-target-naive-100-181903.jpg': 'fd6f1491b2556292d634fce04eeb1133c9b78b82923b541aa071af251e2340c4',
    'lab/src/deltascope/case_studies/images/space_station-target-naive-100-181904.jpg': '53f83763db260dc4cc783a53d3573b90e0bcf349984282480c4a08c29b9c6dd5',
    'lab/src/deltascope/case_studies/images/space_station-target-trigger_amp-200-181901.jpg': '7af5f612dd88b6d4cc8a00a5c2db1246b5687b553be628b6769e1f34c4dd6554',
    'lab/src/deltascope/case_studies/images/space_station-target-trigger_amp-200-181902.jpg': '9b0dccdd8f4c2b009261d0df96e4d4f81c113a6c0cd3e939efc4dd1e2cb6470c',
    'lab/src/deltascope/case_studies/images/space_station-target-trigger_amp-200-181903.jpg': '7e16de2428c07dc0a8ff4fe2b823986c42a8ddf703d46b9344a91e237d8bb848',
    'lab/src/deltascope/case_studies/images/space_station-target-trigger_amp-200-181904.jpg': 'b02f14204d3d0794173101cbf96b79be3f6c28798155e8f35df7e2161366621f',
    'lab/src/deltascope/case_studies/images/space_station-target-trigger_amp-400-181901.jpg': '8925af142925beae8a3c57a82c1fb66a158078611fac7df7ce662f45bf645d8f',
    'lab/src/deltascope/case_studies/images/space_station-target-trigger_amp-400-181902.jpg': '885662f1c0a98765f14327e26107e4ab9cb1cf7caeeeabd84342e3d178b72e31',
    'lab/src/deltascope/case_studies/images/space_station-target-trigger_amp-400-181903.jpg': 'd052ef7b243122386c1632c073d02709f7f71fb61f815c53d2840f0f2d963b13',
    'lab/src/deltascope/case_studies/images/space_station-target-trigger_amp-400-181904.jpg': '3f56bfee10b224f2314fa1780b5dc21086160be29ceaf192a50fcd3f5d37d7f8',
    'lab/src/deltascope/case_studies/images/space_station-target-trigger_amp_peel-100-181901.jpg': 'b864626e211c3b6b7ceeef4ff0e14ab56046efd9fd30e0c32fd82ff3e5cce45f',
    'lab/src/deltascope/case_studies/images/space_station-target-trigger_amp_peel-100-181902.jpg': '014c5a9774d78f88dbd3a96cdf4bcf905dc11340c7a066fed5db8c9ad101e24d',
    'lab/src/deltascope/case_studies/images/space_station-target-trigger_amp_peel-100-181903.jpg': '2876aae5acda789358a5511bb64cdc8767c44b62f5d0a3ccc29db3adac86ce9a',
    'lab/src/deltascope/case_studies/images/space_station-target-trigger_amp_peel-100-181904.jpg': 'fe08d5534c6f1b6a99ce077507b9cec944ff9ebbb419dbf879f4ed6073d8ae73',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-a_side-100-181901.jpg': '4bbb2fd7fbad4b966015f3dca147a5a4611a038a88ba4f54c60d4ca3c1deb162',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-a_side-100-181902.jpg': '96a47a672422da7fd374d85bcbbd0b8e708917bc4869b0c83e1b6d09f204305f',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-a_side-100-181903.jpg': 'f28f8233287a57e4e11d5d6f7ece1d2ad34e6697301afa63df3bede4a6b6e7fb',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-a_side-100-181904.jpg': '81a13b193316fc88a5aa3ea9ca77cff61495e79397005391181f8d26ec4abcdc',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-base-100-181901.jpg': '602c2e38b4ebd099c9be35a442fa84e906b544290087fa2df329e0aa849fea34',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-base-100-181902.jpg': 'ecfc1adbc8fabfe7c2617ade065762b0f7ee7c5856bf1e17b390e3fa8df43fce',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-base-100-181903.jpg': 'd87aa746dce3b45d8678128f9bbe910395163b1734d8454075f78d9f4f0438fe',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-base-100-181904.jpg': '348af9acbf4d50064ab76640f0263a42d8e791b19be219f3ec273b9adf91b341',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-naive-100-181901.jpg': '5f858b73c2e489a9d6549791bfdea9b9a6d1b7a96bde6c28e7dad94669b80150',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-naive-100-181902.jpg': 'd51c863792ee7b93c46ad5106d24d9c0aa14fff974548ef0119d0ad0e34c1a48',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-naive-100-181903.jpg': '037b0f38b7c5eceb9289c5ea8634cd460a9fd77fae6ba72b6c84d0a73c84804d',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-naive-100-181904.jpg': '2d9d0971c769a93567f2ca76ea7cc1b9a2a44c6fd67e8b84c921f7f677c6a501',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-trigger_amp-200-181901.jpg': '5df56860324e9945a6c06a4f4833af4114453bde793ac7f270781713c197e307',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-trigger_amp-200-181902.jpg': '79c518c3bbdae5c80354871d1256b182269c713e8d9180f0b1dcf07c3516d7da',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-trigger_amp-200-181903.jpg': 'd4170cb55b8fa6739b1e769931075d15de4fa403805525b4858634e7a8dd6a51',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-trigger_amp-200-181904.jpg': '50a8618d394cbb6df2ee50bfc2612109c2520feb901aa34929a400b63da94bfb',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-trigger_amp-400-181901.jpg': 'faaff944662c160947c1b435ab4b1e63f38ea3dda54dfc4f785390abf32d1e5f',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-trigger_amp-400-181902.jpg': 'ef70fd194b383c5d8f1849259d6b20dfc1247d53ac074da2e025b61b566eca47',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-trigger_amp-400-181903.jpg': 'd24f6a0003d4357e00b61fbdbdfb8d73362922bbaeb4bc2c122cd892f1fc2310',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-trigger_amp-400-181904.jpg': '97920f139dde7bf838121685d8d4c14f5d0d3d02ad1a972bbeb667791695bdef',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-trigger_amp_peel-100-181901.jpg': 'c95c07e43153ded25a7106592c50410c06339ba9c987ab7c6aa0905c80b78444',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-trigger_amp_peel-100-181902.jpg': '3eb8c17ab529df3116017828a8650d09590c0b4108762093de7255f942d82175',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-trigger_amp_peel-100-181903.jpg': '0b6952dc1f5ecb5840805de771fd4c3bfa9d8d321c33607b8bd7eb31e520bfa7',
    'lab/src/deltascope/case_studies/images/wizard_workshop-generic-trigger_amp_peel-100-181904.jpg': '7c4e324c9dc88e281ca73a5d0bd532b91b9f1ae7609657b29f82fc56606a9dec',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-a_side-100-181901.jpg': '8b27a774d4b2ac261f9730fc67b65fa64eba3561f56b309cbf5438674b0f4cc4',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-a_side-100-181902.jpg': '5d0441cb0038e639557d9a829d18120cbc9af5041fd59acf88817576970e1050',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-a_side-100-181903.jpg': '8ff82fb2ba0edeae087c2fba3ee3c212e6647aadd2fe254654ff648b13c53108',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-a_side-100-181904.jpg': 'f1774f66f37b2c35a6639cf4e76d3dc5a75331802bddcf9361392ff04a139624',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-base-100-181901.jpg': '05809b2aa856d3466ee45b4bf7ff193796790cf5f5cf6a74c601b8260f05e2f7',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-base-100-181902.jpg': 'bfd446be1bbec6b6acef08c545daf318804014925b10181f66939e10a3dded0b',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-base-100-181903.jpg': '94d4c5f28675e96285d7edb5823f983223b579d075e7626d76f8faf5750f123d',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-base-100-181904.jpg': 'd92120093a26650e18fe9fbdaa279750f390be1402402fde1a066da9e4e912cd',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-naive-100-181901.jpg': '6c666ed41d8032c73f678a792e2522acc8856d1c645399c05c42b21058627eea',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-naive-100-181902.jpg': '0e9eff95899c044dbae59688b9bd387bdc14dd0954bf368867100dfa57d58311',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-naive-100-181903.jpg': 'f33d734cb87ceeeb6e033a6cd0472f1472b8b0176ad817eb61db58243a97da18',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-naive-100-181904.jpg': '2b342a35027ba036259f8fe24db08768d056a0bebad0fa284acca152a08b40f7',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-trigger_amp-200-181901.jpg': '2656aa30dff500fca4def97b2f493a712fc917963a8d3d2a79ad93ae0212a9ab',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-trigger_amp-200-181902.jpg': '4984e668cf9c39a8b5723ed8093a89e1ac5a36775483c4e60980153915ebca21',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-trigger_amp-200-181903.jpg': '205e144d6a6999967275101b95a0b4c598a883e7a148db358b0632196a9b80b2',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-trigger_amp-200-181904.jpg': 'f583c9624376840674ccd652ea2133b2158d3d3e5b8b44d21c50f76a4ad04436',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-trigger_amp-400-181901.jpg': '6ae2601d630d8cb100f678f271420914697653646513432ff9fd521b99667061',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-trigger_amp-400-181902.jpg': '16f68d011220f4029cc353efbc6e7d7898b5c3eba4dc0fa92440995c35bab6b4',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-trigger_amp-400-181903.jpg': 'da26b38db1b6f96a481dad9eaadf1fcd9e8796b14c4ab29275ba048597759c6d',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-trigger_amp-400-181904.jpg': '002ec4ba024f38b00baa23b22039e472892b425418c23ea1c90d5fc7de4e9aa4',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-trigger_amp_peel-100-181901.jpg': 'b918bdae2f43235deab5eac2f222fce39ea054b8e7c748884ed7857b774ef2f8',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-trigger_amp_peel-100-181902.jpg': 'd641a25804070e5df56b836bfcb92f4a1d29499655ffecabd9011407dd69e234',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-trigger_amp_peel-100-181903.jpg': 'ce8a078ce2af94277ce45743943cece27266648467fd742026b120f93f2b6b07',
    'lab/src/deltascope/case_studies/images/wizard_workshop-target-trigger_amp_peel-100-181904.jpg': '9955acdf72083da3873df749a47bd4648be2e0fdc5545fe0f93d0ed767c51e89',
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".avif", ".bmp", ".tif", ".tiff"}
MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".onnx", ".gguf", ".ggml", ".h5", ".hdf5", ".npz", ".npy", ".pkl", ".pickle", ".joblib", ".tflite", ".model"}
ARTIFACT_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".log", ".pyc", ".pyo", ".o", ".so", ".dylib", ".dll", ".exe", ".whl", ".egg", ".zip", ".tar", ".tgz", ".gz", ".xz", ".ipynb"}
ARTIFACT_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".cache", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".adapter-review", ".codex", ".agents", ".aws", ".ssh", "build", "dist", "site", "_site", "htmlcov", ".tox", ".nox", "runs", "output", "outputs", "face_crops", "face_crops_hires", "reference_images", "training_data"}
PUBLIC_FILES = {"index.html", "app.js", "styles.css", "dataset.js", "dataset.json", ".nojekyll"}
# A known synthetic privacy-test fixture, restricted to its test module. This is
# not a blanket exemption for tests or Python code.
SYNTHETIC_PATHS = {"lab/tests/test_pipeline.py": {"/" + "private/weights/model.safetensors"}}
URL = re.compile(r"(?:https?|git|ssh)://[^\s<>\"'`]+", re.I)
PRIVATE_PATH = re.compile(r"(?<![\w:/])/(?:home|media|users|private|root|mnt|volumes|var/home|run/user|data/users)/[\w.@$~-][^\s<>\"'`),;\]}]*", re.I)
FILE_URL = re.compile(r"file://(?:/|[A-Za-z0-9_.-]+/)[^\s\"'`<>]+", re.I)
WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]{1,2}[\w.~$-][^\s\"'`<>]*")
UNC_PATH = re.compile(r"(?<!\\)\\{2,4}[\w.-]+\\{1,2}[\w.-]+(?:\\{1,2}[^\s\"'`<>]+)?")
MAX_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class Finding:
    path: str
    reason: str
    line: int | None = None
    source: str = "worktree"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def path_reason(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return "path-outside-release-root"
    if any(part.lower() in ARTIFACT_DIRS or part.endswith((".egg-info", ".dist-info")) for part in path.parts[:-1]):
        return "cache-build-or-private-data-directory"
    if path.name in {".coverage", ".DS_Store", "nohup.out", ".env"} or path.name.startswith(".env."):
        return "local-runtime-artifact"
    suffixes = {s.lower() for s in path.suffixes}
    if suffixes & MODEL_SUFFIXES:
        return "model-weight-or-serialized-array"
    if suffixes & ARTIFACT_SUFFIXES:
        return "cache-database-log-or-build-artifact"
    return None


def scan_paths(name, text):
    # Mask external URLs while keeping line numbers stable. A local file URL is
    # intentionally not masked. Prefix-only detection literals are not paths.
    masked = URL.sub(lambda m: " " * len(m.group()), text)
    for pattern in [PRIVATE_PATH, FILE_URL, WINDOWS_PATH, UNC_PATH]:
        for match in pattern.finditer(masked):
            if match.group() in SYNTHETIC_PATHS.get(name, set()):
                continue
            return masked.count("\n", 0, match.start()) + 1
    return None


def json_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from json_strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from json_strings(item)


def source_path_line(name, text):
    if name.endswith(".py"):
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and scan_paths(name, node.value) is not None:
                    return node.lineno
            for token in tokenize.generate_tokens(StringIO(text).readline):
                if token.type == tokenize.COMMENT and scan_paths(name, token.string) is not None:
                    return token.start[0]
            return None
        except (SyntaxError, tokenize.TokenError):
            pass
    return scan_paths(name, text)


def image_reason(payload, suffix):
    from PIL import Image
    try:
        with Image.open(BytesIO(payload)) as image:
            expected = "JPEG" if suffix in {".jpg", ".jpeg"} else "PNG" if suffix == ".png" else None
            if image.format != expected:
                return "unexpected-image-encoding"
            if image.getexif():
                return "image-exif-metadata"
            if getattr(image, "n_frames", 1) != 1:
                return "unapproved-animated-image"
            allowed = {"jfif", "jfif_version", "jfif_unit", "jfif_density"} if image.format == "JPEG" else set()
            if set(image.info) - allowed:
                return "image-metadata"
            # Unknown JPEG APP blocks can carry metadata that Pillow does not
            # expose through image.info. Only the standard empty-thumbnail JFIF
            # container is permitted; it is required format data, not a caption.
            if image.format == "JPEG":
                for marker, value in image.applist:
                    if marker != "APP0" or not value.startswith(b"JFIF\x00") or len(value) != 14 or value[-2:] != b"\x00\x00":
                        return "image-app-or-comment-metadata"
            image.load()
            if set(image.info) - allowed:
                return "image-metadata"
        with Image.open(BytesIO(payload)) as check:
            check.verify()
    except Exception:
        return "invalid-image"
    return None


def inspect_file(name, payload, *, source="worktree", approved=None, public_hashes=None):
    approved = APPROVED_IMAGES if approved is None else approved
    reason = path_reason(name)
    if reason:
        return [Finding(name, reason, source=source)]
    if len(payload) > MAX_BYTES:
        return [Finding(name, "unexpected-large-release-file", source=source)]
    suffix = PurePosixPath(name).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        allowed = public_hashes.get(name, set()) if public_hashes is not None else {approved[name]} if name in approved else set()
        issues = []
        if not allowed:
            issues.append(Finding(name, "image-not-explicitly-approved", source=source))
        elif digest(payload) not in allowed:
            issues.append(Finding(name, "approved-image-bytes-changed", source=source))
        problem = image_reason(payload, suffix)
        if problem:
            issues.append(Finding(name, problem, source=source))
        return issues
    if public_hashes is not None and name not in PUBLIC_FILES:
        return [Finding(name, "unexpected-public-payload-file", source=source)]
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return [Finding(name, "unapproved-binary-file", source=source)]
    if any(ord(c) < 32 and c not in "\t\n\r" for c in text):
        return [Finding(name, "binary-control-bytes-in-text", source=source)]
    if re.search(r"data:image/[A-Za-z0-9.+-]+(?:;[A-Za-z0-9=.+-]+)*,", text, re.I):
        return [Finding(name, "unapproved-embedded-image", source=source)]
    dataset_js = re.fullmatch(r"\s*window\.DELTASCOPE_DATA\s*=\s*(.*);\s*", text, re.S) if PurePosixPath(name).name == "dataset.js" else None
    if suffix == ".json" or dataset_js:
        try:
            value = json.loads(dataset_js.group(1) if dataset_js else text)
        except json.JSONDecodeError:
            return [Finding(name, "invalid-json", source=source)]
        # Also check decoded strings so JSON escapes cannot obscure a path.
        if any(scan_paths(name, value) is not None for value in json_strings(value)):
            return [Finding(name, "private-absolute-path-in-json", source=source)]
    else:
        line = source_path_line(name, text)
        if line is not None:
            return [Finding(name, "private-absolute-path", line, source)]
    return []


def git(root, *args, input=None):
    result = subprocess.run(["git", "-C", str(root), *args], input=input, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError("Cannot inspect Git release files; run inside the clean repository")
    return result.stdout


def audit_repository(root, approved=None):
    root = Path(root).resolve()
    approved = APPROVED_IMAGES if approved is None else approved
    files = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\x00")
    names = sorted({os.fsdecode(name) for name in files if name})
    findings = []
    for name in names:
        path = root / name
        if any((root / Path(*Path(name).parts[:i])).is_symlink() for i in range(1, len(Path(name).parts) + 1)):
            findings.append(Finding(name, "symlink-in-release"))
        elif not path.resolve().is_relative_to(root):
            findings.append(Finding(name, "path-outside-release-root"))
        elif not path.is_file():
            findings.append(Finding(name, "missing-or-nonregular-release-file"))
        elif path.stat().st_size > MAX_BYTES:
            findings.append(Finding(name, path_reason(name) or "unexpected-large-release-file"))
        else:
            findings.extend(inspect_file(name, path.read_bytes(), approved=approved))
    staged = git(root, "ls-files", "--stage", "-z").split(b"\x00")
    checked = 0
    for entry in staged:
        if not entry:
            continue
        header, name_bytes = entry.split(b"\t", 1)
        mode, oid, stage = header.decode("ascii").split()
        name = os.fsdecode(name_bytes)
        checked += 1
        if stage != "0" or mode not in {"100644", "100755"}:
            findings.append(Finding(name, "unmerged-symlink-or-gitlink-entry", source="index"))
            continue
        reason = path_reason(name)
        size = int(git(root, "cat-file", "-s", oid))
        if reason or size > MAX_BYTES:
            findings.append(Finding(name, reason or "unexpected-large-release-file", source="index"))
            continue
        findings.extend(inspect_file(name, git(root, "cat-file", "blob", oid), source="index", approved=approved))
    return findings, {"worktree_files": len(names), "index_blobs": checked}


def public_image_hashes(root, approved):
    """Derive the export bytes from approved sources with the installed encoder.

    The source hash is checked first. This supports the exporter's metadata-free
    re-encoding without approving arbitrary replacement pixels or filenames.
    """
    from PIL import Image, ImageOps
    allowed = {}
    for name, expected in approved.items():
        path = root / name
        if not path.is_file() or path.is_symlink():
            continue
        raw = path.read_bytes()
        if digest(raw) != expected or image_reason(raw, path.suffix.lower()):
            continue
        suffix = ".png" if path.suffix.lower() == ".png" else ".jpg"
        stream = BytesIO()
        with Image.open(BytesIO(raw)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if suffix == ".png":
                image.save(stream, format="PNG")
            else:
                image.save(stream, format="JPEG", quality=95, optimize=True)
        allowed["images/" + expected + suffix] = {expected, digest(stream.getvalue())}
    return allowed


def audit_public(root, folder, approved=None):
    approved = APPROVED_IMAGES if approved is None else approved
    root, folder = Path(root).resolve(), Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError("Public payload directory does not exist")
    allowed = public_image_hashes(root, approved)
    findings, count = [], 0
    for directory, dirs, files in os.walk(folder, followlinks=False):
        for name in list(dirs):
            path = Path(directory) / name
            if path.is_symlink():
                findings.append(Finding(path.relative_to(folder).as_posix(), "symlink-in-public-payload", source="public"))
                dirs.remove(name)
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(folder).as_posix()
            count += 1
            if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
                findings.append(Finding(relative, "nonregular-public-payload-file", source="public"))
            elif path.stat().st_size > MAX_BYTES:
                findings.append(Finding(relative, path_reason(relative) or "unexpected-large-release-file", source="public"))
            else:
                findings.extend(inspect_file(relative, path.read_bytes(), source="public", approved=approved, public_hashes=allowed))
    for required in ["index.html", "dataset.json", "dataset.js"]:
        if not (folder / required).is_file():
            findings.append(Finding(required, "missing-public-entry-file", source="public"))
    return findings, {"public_files": count}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--public-dir", type=Path, help="Also inspect every file in a generated static export")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        findings, counts = audit_repository(args.root)
        if args.public_dir is not None:
            extra, public_counts = audit_public(args.root, args.public_dir)
            findings.extend(extra); counts.update(public_counts)
    except (RuntimeError, ValueError, ImportError) as error:
        print("Release audit could not run: " + str(error), file=sys.stderr)
        return 2
    # The same unsafe file can exist in both the index and worktree; retain both
    # findings so an unstaged cleanup cannot conceal staged release contents.
    report = {"ok": not findings, **counts, "approved_image_count": len(APPROVED_IMAGES),
              "findings": [asdict(item) for item in findings]}
    if args.json:
        print(json.dumps(report, indent=2))
    elif findings:
        print(f"Release audit failed: {len(findings)} finding(s).")
        for item in findings:
            location = item.path + (f":{item.line}" if item.line is not None else "")
            print(f"  {item.source}: {location}: {item.reason}")
    else:
        print("Release audit passed: " + ", ".join(f"{value} {key.replace('_', ' ')}" for key, value in counts.items()) + ".")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
