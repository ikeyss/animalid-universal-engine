---
title: AnimalID Universal Engine
emoji: 🐾
colorFrom: green
colorTo: blue
sdk: gradio
app_port: 7860
---

# AnimalID Universal Engine

Backend isolado do aplicativo AnimalID para identificação bioacústica universal.

## API

- `GET /health`
- `POST /identify-animal` com campo multipart `audio`

Exemplo de resposta: `identified`, `animal`, `group`, `confidence`, `confidence_percent`, `candidates`, `model` e `requires_confirmation`.

O contrato retorna grupo, espécie, confiança, candidatos, modelo e necessidade de confirmação. O backend não usa BirdNET como se fosse uma base universal.

O aplicativo decide a rota antes de aceitar o resultado: mamíferos, anfíbios, insetos e répteis são priorizados pelo classificador amplo; aves seguem para BirdNET. Resultado de ave sem evidência acústica independente é recusado para evitar falsos positivos como cachorro identificado como tiziu.

## Modelo

Para o plano gratuito do Render, o motor usa YAMNet TFLite em CPU, um modelo de 521 classes AudioSet. Ele aceita somente classes biológicas: animais, insetos, aves, mamíferos, anfíbios, répteis e vida marinha. Voz humana, música, veículos, chuva e outros ruídos são descartados. Confiança abaixo de 50% é recusada; resultados abaixo de 70% retornam `requires_confirmation=true`.

O modelo geral fornece classes de som, não garante espécie científica. Pesos e tokens devem ser Secrets/Volumes do Hugging Face, nunca gravados no APK.

## Deploy automático

## Deploy gratuito no Render

O arquivo `render.yaml` configura um Web Service gratuito, executando diretamente `main:app`. Para automatizar tudo, execute `deploy_animalid_tudo.bat`, informe o usuário GitHub e uma Render API Key; o script autoriza o GitHub CLI pelo navegador, cria/publica o repositório e cria ou redeploya o serviço pela API oficial. Depois use a URL `https://<servico>.onrender.com` em `ANIMAL_ENGINE_URL`. O plano gratuito dorme após 15 minutos sem tráfego e tem 512 MB de RAM; o motor foi reduzido para YAMNet TFLite para caber nesse limite.

Após a primeira publicação, use `redeploy_animalid_render.bat`. Esse script executa somente o redeploy do serviço já existente e exige apenas a Render API Key.

O comando único recomendado é `deploy_animalid.bat`: ele detecta automaticamente se é a primeira publicação. Para publicar alterações locais antes do redeploy, use `deploy_animalid.bat -SincronizarCodigo`.
