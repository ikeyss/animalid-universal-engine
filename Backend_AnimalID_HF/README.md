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

## Modelo

Para o plano gratuito do Render, o motor usa YAMNet TFLite em CPU, um modelo pequeno de 521 classes AudioSet. Ele reconhece apenas classes sonoras que o modelo realmente separa; não inventa espécie quando há apenas a classe genérica “animal”. Confiança abaixo de 50% é recusada e entre 50% e 69% retorna `requires_confirmation=true`.

O modelo geral fornece classes de som, não garante espécie científica. Pesos e tokens devem ser Secrets/Volumes do Hugging Face, nunca gravados no APK.

## Deploy automático

## Deploy gratuito no Render

O arquivo `render.yaml` configura um Web Service gratuito, executando diretamente `main:app`. Para automatizar tudo, execute `deploy_animalid_tudo.bat`, informe o usuário GitHub e uma Render API Key; o script autoriza o GitHub CLI pelo navegador, cria/publica o repositório e cria ou redeploya o serviço pela API oficial. Depois use a URL `https://<servico>.onrender.com` em `ANIMAL_ENGINE_URL`. O plano gratuito dorme após 15 minutos sem tráfego e tem 512 MB de RAM; o motor foi reduzido para YAMNet TFLite para caber nesse limite.
