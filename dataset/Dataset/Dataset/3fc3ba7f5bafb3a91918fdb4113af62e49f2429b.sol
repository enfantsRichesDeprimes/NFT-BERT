/**
 *Submitted for verification at Etherscan.io on 2021-12-04
*/

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// cryptohoots
contract characterImageSender {
    string public constant characterModel1 = "iVBORw0KGgoAAAANSUhEUgAAAFAAAABQBAMAAAB8P++eAAAAG3RFWHRTb2Z0d2FyZQBDZWxzeXMgU3R1ZGlvIFRvb2zBp+F8AAAACXBIWXMAAC4jAAAuIwF4pT92AAAAElBMVEVxcXFOTk5AQEAAAADFxcX////noDnkAAADfklEQVRIx+XWQZLjuA4E0MTQvQeCrgtQFwCV5F4KUfuqoHn/q8zCdrdrSnL//ffSegFKBMFIpP/xh/97OM3fn008g/wP7GdwfC/JU9j6q2Q9hWXvfHHtDKay2uCcUppIVtvnM1ibGcg51TG+7A1kM5jZPNVhptdxDseqACLNDHgDE4cBsKwA5Mo3LaQDkKwAjPM5nEb3Bwwc7+BtjQ7JisD2DvK2SnTJYLY23rxjbQqJJAG57v0tJCVmBPpfYIS7ZHgI9hcY3MXhDNLewTUCLhmOv0ElgayB/hZOTQGQWQHZ3u1jNUXISiqknEPWbghUGGlUnp2eqXYjmRWWI43jdFxHz2ZGRcxm6vvZzNTRNzGLimgAt8vrUOJ15VE2ANFAU+Tm+xnsrRGwLNkQpG07j+HobaVDGE0R83W79ENYxtZiUEiMBil+bWU7hrfQ4upApELKYs1P4FJaWP0+hZI3eQPXsBIQA4JtuJ7DhWsEcIf56uEYhrb4GgEIEKy5eJgP4NRCQ5EIAAD16rbxCDpCg0uEAkDWa9DwcpW+QP0OZbPwcQSJCLhEGCDIKpvwsGLxALhEuUNDYM5/rpUXSCLqA2o2M37kXwcwxw3RLJopzJgt5p0fP+G0cmE2i2pqZplk3rsewbzkbJZVzEyyGWXv2n9Cq0uO5AO6WZDr8AO41iWbWVaYmURD8Ov40+0/UOqa5QFVoijc9jMIs6wQhURRBNt5BkWzAgCiGYLt/QB6XbP+huaGkK8HMHldq0p8geKn0CTa/TwKTc2v/aCFU13rl0XicXLFPvplP4Kfa/0yWgYAiSFb6ziCKVv9sqxBAYiDtnf/AZlSomaNX0AGEICgbbzMK55JhmkCI9uCAICA8LL7z+GayMkDbxsRADguK9GOpnAaJLcb60aS3HsjcXjtcbDw5nXDGAOlF/ZHQfL79pSdvHnZ2Fp7wEeF/oDzUzJ0Lx7Y2lJYt8o5pcQynhXJe2ecF16Kt0ayePVQeI9X8wNOlSklAn7hr+KttbYULyWULSU+kgAe3zHx89I6mQrrGGMQLMHb/Cz4+BiO+nnftXmaSZL8hyV8Xsg2+sv2TPWG/XvMnGbiF7b6zCDPXvv37Hj/c0G7PVZ+Lv15+enSVIDnync4DWzzQVKeFvxOP0gppbr84mGmLsvvM477u/Tj8D01vMCpHi+cUprK8nz0L3y7sq+v6BBeAAAAAElFTkSuQmCC";
    string public constant characterModel2 = "iVBORw0KGgoAAAANSUhEUgAAAFAAAABQBAMAAAB8P++eAAAAG3RFWHRTb2Z0d2FyZQBDZWxzeXMgU3R1ZGlvIFRvb2zBp+F8AAAACXBIWXMAAC4jAAAuIwF4pT92AAAAFVBMVEXFxcVxcXFOTk5AQEAAAAD///9gYGBzK4WWAAADw0lEQVRIx53WwXLbIBAGYKyRchaU3K0d8gTW5AxoucsZuGemnbz/I/QHS64dg9upp5Me8mVhl12MOP3jR/wHfP/6+vkv8OtTiJc2vcIvUT5NucN3sX3+Bj93+PIcloC9vQ858yMsAXu+DznHB3jZYZ/sHeQmFNHerM3hEV5S6e0Q+RqSg2hBYTtKbF9yGsyBhqUFgyCSzMsppPRBFbhXmwVJIlrmkIimY2rD1E1SSsPQUj6BkACSfObqyKcmFOwyKZB4acM+RbdBzekZXDvjCtQsKvDaO7x2yjjlJXsSidswiEkqg2rnXIb4FDIr47Gyq8L3KzRpyEuHNdJf4AroJBpCiQq8HnZnUioRUwO+73DKifgJe6zDvSNFPhPmXHHVL+1xRROi1h5hUSa7NC8AND+hMhPOmYknXhpXSg/HZV3yaDWduHFJ/UqrRyciosH/EzJvXHshrb0iMoBoSx5SXKpwzjA3OOYGOfkhpQZMaRWcG1x5WCXS7Sa/wQ71VmxoYpNhrELMqCBt0D2GtPfyODyB2JrBFE6ansK1kxliCjMch9S3ITYoFY4HNVJirUObVvxWG+Qt+RnkoUxMgR5QHoVeKnDWoivldhIpcYYHrkGr7A30Bd5cpTeQGHtzO9QsSb9VYd5dAPRyQr11RAXqEc3kCsyh0JARre7/tMU3yN6QuUCW/OZfK9CbLk+KJuPIE/qc/cBvj3DuuGOPaZkoQ4+RwF0x1aDvPJPxU24zdCSxAoyPkMIBG8OSuKRIOSKtjslVYAeI0SozSArzoB2gfoQqdF5tEFHVJB0NLSgLxG1WoKaBW1BN+ZBdPnLKMFagA5yuEDXCyRwr8AQYcsZ/oHJNWLLNe0RDECp/jJUjnAE/KH8lcPmh6C2OQw2eM8R1R/nDGt80UdbgyRMgJjWXHd9fjJzdA8yXDCNj84FrDAOh8W8SyfXfGxdvg9OMIWVxyChjxePgHocLcnaa154LdHLsWIraFM4JmfYrhz6nzEMUuNiq1x7jgcKrC71MKUkbLcctIPN9eezAgLY8vDa4RYgbXHbJOjqLnQpxsNhDKBHZpj1iCY2TcTzyaB2+b5itC05bLguVcxSFlDIi15FfM8wRnbXaooy8vQTElsfM51FExMaaSCYhY6udWPaAWzKcwvlStWVeSnn4B+B5ZASMN+WZwyqHu7fGCX8gX2Uf9jfIftZOxOX7s5YPUqzbyvvS5/HRIWEp95UvcE6yXyov5Rkh92+vAsPhlatvanu49ri47CXWH9+zkDdwDvWFyy4P+69+AzD+OQLjEEHXAAAAAElFTkSuQmCC";
    string public constant characterModel3 = "iVBORw0KGgoAAAANSUhEUgAAAFAAAABQBAMAAAB8P++eAAAAG3RFWHRTb2Z0d2FyZQBDZWxzeXMgU3R1ZGlvIFRvb2zBp+F8AAAACXBIWXMAAC4jAAAuIwF4pT92AAAAJFBMVEXFxcVxcXFAQEAAAABOTk7///8nJyc5OTmamppgYGBCQkLp6ekOUXVPAAAEmUlEQVRIx5XWMW/jNhQAYBqyh2xmJR+M3GZN3kg9ST5clxPEoaMNHQMj28EAbbhTYMQabgoKJcBNB9+U0QjqC7QWbYP6z/WRtlI7ptJWCZAg/vLI90Q+kvT+40P+B+zvf+3f369ehV93H99/JeSsnpJen5wZR8xTK6uIfbJ/Vv8yx68VPHs9axPQYcchfTiFJqADxyH97ATuZuhIdgTBwJUFkowdjA2Rhv2zQ7hLxWGtDJ5DQkQy89HqBBLWFBLYmU4DIBKtoCYiiYgQFCDoRVJ+FDt4NMeq2kAEFUIEfiSFSD7I4GXWz1A2E0ppCqgpfQWiREBFqLn3AXq1kADXxEABQT10ZMb30AX5Ghw2U26gC8QCn9cODJteyr2QQiiIhHoYkYR6KVZb59LKXoUAXhriyNwK+88wlS09dDTMRD10NBwi5BQXhEcyv2bhchY1UylNRKkhD6xbwaEIE51ImOAc7VCHROgQ/U4AdMU9B6h7CvsaurgIsdYhhsUyMW6D2ACcLo0ygTrB9wwCEn5jg737b8WNi0KPK0Jcau5sbIU9Xtych7gSMWKKP5P4sR7SB0+IFCEuS5h3a+C78ZhO9ALHfYM5xfOiuAks0H9EqABR6IWCgyqL4osVrnGOmwkMQkhFCqkqu+PzWqgmabxI03QQxtP5Yx3s0lKpMF5giZYarm9KKywp3UwR5oPBAGE+X9OJLWt/Ts9VriBW+gum+XJ9bmD/ZXne0xI/nSACiDUs3zyY9bJ6AeNyYyBwQTXM55tFYIF+MVc5QtyvCYUYoZpug151apDndsW7FwjzidAw3MEn9yBi1QD5o1Q5DCq4gFx9vnMPkqlaKi+KPI8QhjRxEWb5cvvkHp5cVcRi9qeBQIUr4kmWX//1dBUc1HF1ACGeiHQHId9+P4ZVGYvZFsuzECkXofjxj6m8/N65OYV+V862F/nyt0Ro+H77+3izfntngxezS6l+DhMPl5kXl+X2FqF7Ch9Hs4tf5RYh7hgvVur6dk7HFthFOC9LsweFl6r88xjhwg6XuB41xKhqmj+pdR3MDcRuZuD17brzSw3Mp7rtYMNPlaqD7xBOlxUUAzNJC+whvJwuFwdwWQ8Vvhiq50g9UFOFaVteoT+ajTZKb0HQ5wIs1afOl7UNvtEQ253QD7ixyjt3NtiLZ6NzrI6rOxruB1BXb8cnUB9S8NPFp4eP2MawUbn4Pc3p+OHlvsa7gV4/EyANjTT2IL8an3YKlD4vYKg7PNUFajcRZpaW4kvM1BlC5OiUoZURgKvM1lIALygw5JFDpZSUZQyy/dEOcFwe1gKEzFy89nAfIdvDoJLgZpxxPNNJg+EcIhMRmKwimtA9P+LQhjbjeL4DMB5xl4EZyKREDDFlxFzb0NFQR+SMuczBP+9vAmSfhw+jNskwNo6JyUhs+szlJKgC7pMBGY0o0VMK/MCUB35AOGoDBswOyuNHQ9o6umv08B9ohzpRdQep3jUnWfDyWgsNSob7kauhR+1ThwlTWo28g76kTmC5KfsYsrr9GBg1OmC9U7NGuxX8A3Eumf3y7RN6AP3IPrCZZaP66G8U6z/914yl/AAAAABJRU5ErkJggg==";
    string public constant characterModel4 = "iVBORw0KGgoAAAANSUhEUgAAAFAAAABQBAMAAAB8P++eAAAAG3RFWHRTb2Z0d2FyZQBDZWxzeXMgU3R1ZGlvIFRvb2zBp+F8AAAACXBIWXMAAC4jAAAuIwF4pT92AAAAJFBMVEXFxcVxcXFAQEAAAAAnJyf///9OTk45OTlgYGCamppCQkLp6elzSOZsAAAFFElEQVRIx5XWzW/bNhQAcAaCixzNqRLg9FTr4qOoJ8podyrMQbB3scCogHuaJoDAjoEV8NBLkF1yDVajQG9GMhTxdRuGzP/cHiXLX5KLjXHsIPrp8ZEmn0j6/7GR/wEHmz8Hd3e334Q31eW7G0LOT1PSH5Dz0hFCfEJqWX4ELREH6CyJbxUcmJs92ZLjjYEphjzfReSyOWoTkFiOvw1pGreb8GYHz78Fy4AIZQu8bYFU7vcNqV0PawtvKji8lrANCam2y0u3DUiGxQ8p+CUESIUOTkQkTAuRQtnSNBZXwXGOg+pbIaCFLtKyCaHjEgb9IwgogRZxIYS4xF+tY4rIa8AoL6XQNSxiQBc1oJWZuYGhLjZQABqWNeF7KsFKpZGX2PEc0qAPrt0/hiTKnBSmL/IhhozRkTTwkuwQltPDlOMgdPMh9guRIilwmskm5G5GOcmomxdiTqn7U0cmLg2a0EqUgQBuKJgNDCH92Qmaq8dK3IyTPO0whHwqVUeq13bbMuO2QjitIJEukflr2bZwOYv4izxNmT2iPDWQsQN4V/dNEWKO1B5lmKOBPDjYrrdVSIQ5jppSgAg/XAtvCY4jmpAWtYGrjNoRhsVp8tkRrNodsXqUS4U6owpAQcbu7bYidffr4t5GYfpVUQ7KHs9aYZ8t7i8ipTAUzfEzC59OQ/rgKpUjVDikSe8EfDOb0REOGhXgmMLJYnHfMpi+94RQAKLIjRQDsVosvrTCJeb4OIJhBLnKIRer3uziJBSjPJzneT6MwnjydAr26EqIKJzjFBUGLu9XrXBF6WOMUA+HQ4R6sqSjtlF7E3ph9l4ozA/EulhelHBwPD1v6QqvjhABhAauXj6US/D2CIarxxICU9RAPXmcBy3QW0xwOxejnNGMQohQxOugXz81yLZcsZ7Z93qkDIwq+GzvRawLIHvCigjDGs6xtH38bO8Npi6pbLHQmiOMaGYjlLpYP+/vwm3ExfjvEgJVtgpHUl//83y1v69v9yCEI5VXEPT66yGsp3ExXuP0zFXOVKS+/zNOP3x17pvQ66Xj9aUu/siUgW/Xf80el68+t8HL8YdU/BhlLi4zN1yt1p8Q2k34lIwvf0/XCHHHYKES158mdNYCewgnq1W5B5WbC/1xhnDeDrHYlhCjYtV9FssWCAbqErqMlvD609L57QTUsSk7DLci1udW6Mk3COOihmpYJtkCE4Qf4mK+B4s2yF0DBX4x1ORIXRCxwGFD/djcQS8ZJ4/CbEHIzVsh3jlfllhKB/XCNXd4CbVfGojlTpkGdii083nJgl1Ec4eXvIZwnFzg7NimouF+AHH1albC/l5E7sg+/HL57uE9ljEsVDa+Yk1nD9tH1yZHBmDWzwjImUEGu6CvZrunZgU98AA8toCpqfDUTFD3BUIJzmFED1/mMGFNgVvlmaIjCcAV5gONFQ54QIEp4xbFIwX1pQ8SqpGUeK8A+B1A6Ft49CEbuIkgNzCoJdiS+fhkJeTMxxx4GRH8tI5Y5eFxBl3o+gxPFgA+48z2oexIBhvocSNxRlgXHANNROb7tm/hv0kZsMoRx+FB0iUSY2Of5oCERd+3GQnqgJvBQMoTSkxKgRdUZ67vECZdwIByfx75lHbg4JSMN1CHWnzTcz09wIgMjk/UcEbJdNNz3XXSbTocMKV1zxX0UmoFLYd0D0Nueq4gP3Ogv3dW3Db/rNsJdhBzkbtj+kFIQvegx7cd3zayPKsv/Qs7TGLvPxOTKAAAAABJRU5ErkJggg==";
    string public constant characterModel5 = "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";
}