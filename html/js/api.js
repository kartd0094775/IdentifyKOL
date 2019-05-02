const generalOptions = (data) => {
    return {
        headers: {'Content-Type': 'application/json'},
        method: 'POST',
        body: JSON.stringify(data)
    };
};

const fetch_similar_word = (data) => {
    let requestOptions = generalOptions(data);
    return fetch('/api/query/similar/', requestOptions).then(
        response => {
            if (!response.ok) {
                return Promise.reject(response.statusText);
            }
            return response.json();
        }
    );
};
const fetch_object_posts = (data) => {
    let requestOptions = generalOptions(data);
    return fetch ('/api/scape/object/', requestOptions).then(
        response => {
            if (!response.ok) {
                return Promise.reject(response.statusText);
            }
            return response.json();
        }
    )
}

const fetch_rank_facebook = (data) => {
    let requestOptions = generalOptions(data);
    return fetch('/api/query/fb/', requestOptions).then(
        response => {
                 if (!response.ok) {
                     return Promise.reject(response.statusText)
                 }
                return response.json();
        }
    ).catch(
        message => {
            console.log(message)
        }
    )
};

const fetch_scape_facebook = (data) => {
    let requestOptions = generalOptions(data);
    return fetch('/api/scape/fb/', requestOptions).then(
        response => {
            if (!response.ok) {
                return Promise.reject(response.statusText)
            }
            return response.json();
        }
    ).catch(
        message => {
            console.log(message)
        }
    )
}

const fetch_scape_google = (data) => {
    let requestOptions = generalOptions(data);
    return fetch('/api/scape/google/', requestOptions).then(
        response => {
            if (!response.ok) {
                return Promise.reject(response.statusText)
            }
            return response.json()
        }
    ).catch(
        message => {
            console.log(message)
        }
    )
}
