// Copyright (C) 2022 Intel Corporation
// Copyright (C) 2023 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import Axios from 'axios';

import './axios-config';

onmessage = (e) => {
    const config = { ...e.data.config };
    if (config.removeAuthHeader) {
        delete config.removeAuthHeader;
        config.transformRequest = (data, headers) => {
            delete headers.common.Authorization;
            return data;
        };
    }
    Axios.get(e.data.url, config)
        .then((response) => {
            postMessage({
                responseData: response.data,
                headers: response.headers,
                id: e.data.id,
                isSuccess: true,
            });
        })
        .catch((error) => {
            postMessage({
                id: e.data.id,
                message: error.response?.data instanceof ArrayBuffer ?
                    new TextDecoder().decode(error.response.data) : error.message,
                code: error.response?.status || 0,
                isSuccess: false,
            });
        });
};
