import json
import logging
import os
import sys

import requests
from dateutil import parser


class ArgumentsError(Exception):
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors


class GitHub(object):
    def __init__(self, api_url, token, session_verify=True, proxies=None):
        self.proxies = proxies
        self.api_url = api_url
        self.token = token
        self.session_verify = session_verify

    def _iterage_resource(self, url, params):
        url = os.path.join(self.api_url, url)
        another_page = True

        with requests.Session() as session:
            session.headers['Authorization'] = 'token {}'.format(self.token)
            session.verify = self.session_verify

            while another_page:
                r = session.get(
                    url, params={**{'per_page': 100}, **params}, proxies=self.proxies)
                if not r.ok:
                    logging.error(
                        "Can't get resource {}\nStatus code: {}".format(url, r.status_code))
                    sys.exit(1)
                json_response = json.loads(r.content)
                if isinstance(json_response, dict):
                    yield json_response
                    another_page = False
                elif isinstance(json_response, list):
                    if 'next' in r.links:
                        url = r.links['next']['url']
                    else:
                        another_page = False
                    for el in json_response:
                        yield el
                else:
                    return

    def release(self, org, repo, tag_name=None, release_id=None, params={}):
        if tag_name is None and release_id is None:
            logging.info("tag_name or release_id must be specified")
            return

        tail = release_id if tag_name is None else "tags/{}".format(
            tag_name)

        return next(self._iterage_resource("repos/{}/{}/releases/{}".format(org, repo, tail), params))

    def releases(self, org, repo, start_date=None, end_date=None, params={}):
        for r in self._iterage_resource("repos/{}/{}/releases".format(org, repo), params):
            published_at = parser.parse(r["published_at"])

            if start_date and published_at <= start_date:
                break
            if end_date and published_at > end_date:
                continue
            yield r

    def repositories(self, user=None, org=None, params={}):
        if user is None and org is None:
            raise ArgumentsError("'user' or 'org' must be specified.")

        unit = "users" if user is not None else "orgs"
        for r in self._iterage_resource("{}/{}/repos".format(unit, user if user else org), params):
            yield r

    def languages(self, owner, repo):
        url = os.path.join(
            self.api_url, "repos/{}/{}/languages".format(owner, repo))

        with requests.Session() as session:
            session.headers['Authorization'] = 'token {}'.format(self.token)
            session.verify = self.session_verify

            r = session.get(url, proxies=self.proxies)
            if r.ok:
                return json.loads(r.content)
            else:
                logging.error(
                    "Can't get resource {}\nStatus code: {}".format(url, r.status_code))
                sys.exit(1)

    def paths(self, owner, repo, depth=0, sha="master"):
        url = os.path.join(
            self.api_url, "repos/{}/{}/git/trees/{}".format(owner, repo, sha))

        def _paths(url, pred, depth):
            r = session.get(url, proxies=self.proxies)

            if not r.ok:
                logging.error(
                    "Can't get resource {}\nStatus code: {}".format(url, r.status_code))
                sys.exit(1)

            data = json.loads(r.content)
            files = [os.path.join(pred, el["path"]) for el in data["tree"] if el["type"] == "blob"]

            if depth == 0:
                return files

            for el in data["tree"]:
                if el["type"] == "tree":
                    files += _paths(el["url"], os.path.join(pred, el["path"]), depth - 1)

            return files

        with requests.Session() as session:
            session.headers['Authorization'] = 'token {}'.format(self.token)
            session.verify = self.session_verify

            return _paths(url, "", depth)

    def file(self, org, repo, file_name):
        url = os.path.join(
            self.api_url, "repos/{}/{}/contents/{}".format(org, repo, file_name))

        with requests.Session() as session:
            session.headers['Authorization'] = 'token {}'.format(self.token)
            session.verify = self.session_verify

            r = session.get(url, proxies=self.proxies)
            if r.ok:
                return r.content
