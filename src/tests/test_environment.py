import sys


def main():
    system_major = sys.version_info.major
    system_minor = sys.version_info.minor
    if system_major != 3 and system_minor != 9:
        raise TypeError(
            "This project requires Python 3.9. Found: Python {}".format(
                sys.version
                )
        )
    else:
        if "zoidberg_env" not in sys.prefix:
            raise TypeError(
                "This project requires 'zoidberg_env' environment. Activate it"
            )
        print(">>> Environment checked ")


if __name__ == '__main__':
    main()
